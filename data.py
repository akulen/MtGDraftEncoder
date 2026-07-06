import os

from gpu_management import set_gpus

if __name__ == "__main__":
    set_gpus(1, forcing=True)
    print(f"XLA_PYTHON_CLIENT_ALLOCATOR set to: {os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')}")
    print(f"CUDA_VISIBLE_DEVICES set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

from functools import cache, partial
import hashlib
import datetime
import humanfriendly
import json
import numpy as np
import pickle
import polars as pl
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, List, Tuple
import jax
import jax.numpy as jnp
from beartype import beartype as typechecker
from jaxtyping import jaxtyped, Array, Bool, Float, Int, PRNGKeyArray
from typing import Dict, Literal, Mapping, Optional
import equinox as eqx

import time

from models import DraftWRPredictor
from nlp import Gemma, NLPProcessor
from data_types import Cards, Sets, Drafts
from card_utils import (
    get_all_cards, make_oracle, fix_split_names, get_draft_data, get_card_stats
)
from training import Trainer

os.makedirs('data', exist_ok=True)
CSI = "\x1b[" #]

@cache
def get_set_config() -> Mapping[str, Mapping[str, Any]]:
    with open('data/set_config.json', 'r') as file:
        return json.load(file)

class DL17Lands(Dataset):
    def __init__(self, format='OTJ', include_ext=False, verbose=1):
        st = time.time()
        self.format = format
        self.all_cards = get_all_cards()
        self.verbose = verbose
        if verbose > 0:
            print(f"Loading {format}")
            print('-' * 11)
            print(f"Loading card data: {time.time() - st:.6f}s")
            st = time.time()

        try:
            set_config = get_set_config()[format]
        except KeyError:
            raise NotImplementedError(
                f"Format '{format}' is not (yet) implemented."
            )
        self.cards, self.drafts = self.collect_format(
            **set_config, include_ext=include_ext
        )
        self.pack_size = set_config.get('pack_size', 15)
        if verbose > 0:
            print(f"Making extension dataframe: {time.time() - st:.6f}s")
            print('#' * 50)

    def collect_format(
        self, stlands, expansions=None, exclude=None, special_guests=None,
        thelist=None, include_ext=False, pack_size=None, pad_id=0,
        data_exclude=None, cube=None
    ):
        if exclude is None:
            exclude = []
        if special_guests is None:
            special_guests = []
        if thelist is None:
            thelist = []
        assert(len(self.all_cards.filter(pl.col('id') == pad_id)) == 0)

        if cube is not None:
            df_cards = self.all_cards
        elif expansions is not None:
            # We drop cards that only appear in set or collector boosters
            df_cards = self.all_cards.filter(pl.col('is_booster') == True).filter(
                (
                    pl.col('expansion').is_in(expansions)
                    & ~pl.col('name').is_in(exclude)
                )
                | (
                    (pl.col('expansion') == 'SPG')
                    &  pl.col('name').is_in(special_guests)
                )
                | (
                    pl.col('name').is_in(thelist)
                )
            )
        else:
            raise ValueError(f'Either cube or expansions must be defined [{stlands}].')
        # Split cards are listed 3 times (once fully, then once for each half).
        # so we drop the half cards, then rename the full to keep only the
        # first half
        split_cards = (
            df_cards.filter(pl.col('name').str.contains('//'))['name']
            .str.split(by=' // ')
        ).explode()
        df_cards = df_cards.filter(
            ~pl.col('name').is_in(split_cards.implode())
        )
        df_cards = df_cards.with_columns(
            name=pl.col('name').str.split(by=' // ').list.get(0)
        )
        # Drop duplicate basics
        # Stopped the assert as list cards can be duplicates
        # dup_rarities = df_cards.filter(
        #     pl.col('name').is_duplicated()
        # )['rarity'].unique()
        # assert(len(dup_rarities) == 1 and dup_rarities[0] == 'basic')
        df_cards = df_cards.unique(subset='name')

        oracle = make_oracle(df_cards)
        df_cards = df_cards.with_columns(
            oracle=oracle['oracle'],
            full_name=oracle['name']
        )
        df_cards = fix_split_names(df_cards, stlands, verbose=self.verbose)

        df_drafts = get_draft_data(df_cards, stlands, pack_size, pad_id)

        df_cards = get_card_stats(df_cards, stlands, include_ext, data_exclude)

        df_cards = df_cards.sort(by="id")
        df_drafts = df_drafts.sort(by="draft_id")

        return df_cards, df_drafts


def collate_drafts(
    set_id: int,
    dl: DL17Lands,
    time_offset: int=0,
    time_window: int=7,
    first_drafts: bool=False,
    first_n: int=-1,
    pad_id: int=0,
    verbose: int=0,
) -> Drafts:
    cache = f'drafts_{dl.format}'
    if time_offset > 0:
        cache += f'_skip-{time_offset}'
    cache += f'_{time_window}-days'
    if first_n > 0:
        cache += f'_first-{first_n}'
    if pad_id != 0:
        cache += f'_pad-{pad_id}'
    if first_drafts:
        cache += '_reversed'
    if os.path.exists(f'cache/{cache}.pickle'):
        drafts = pickle.load(open(f'cache/{cache}.pickle', 'rb'))
        drafts = eqx.tree_at(
            lambda x: x.set_id,
            drafts,
            jnp.full(drafts.set_id.shape, set_id, dtype=jnp.int16)
        )
        return drafts
    assert(len(dl.all_cards.filter(pl.col('id') == pad_id)) == 0)
    PERIOD = datetime.timedelta(days=time_window)
    draft_start = (dl.drafts
        .select(pl.col('draft_time'))
        .min()
        .collect()['draft_time']
        .str.to_datetime()[0]
    )
    last_draft = (dl.drafts
        .select(pl.col('draft_time'))
        .max()
        .collect()['draft_time']
        .str.to_datetime()[0]
    )
    first_draft = last_draft - PERIOD
    if verbose > 0:
        print(
            f"All drafts:         {CSI}31m{draft_start}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
    OFFSET = datetime.timedelta(days=time_offset)
    first_draft = first_draft - OFFSET
    last_draft = last_draft - OFFSET
    if first_drafts:
        first_draft = draft_start + OFFSET
        last_draft = first_draft + PERIOD
    if verbose > 0:
        print(
            f"Picking drafts from {CSI}34m{first_draft}{CSI}0m"
            f" to {CSI}32m{last_draft}{CSI}0m"
        )
    if draft_start > first_draft + datetime.timedelta(days=1):
        print(f'{CSI}33mWarning: {CSI}34mDraft period starts ({CSI}32m{first_draft}{CSI}34m) significantly earlier than first draft ({CSI}32m{draft_start}{CSI}34m).{CSI}0m')
    n_picks = dl.pack_size * 3
    rank_dict = {
        'null': 0,
        'bronze': 1,
        'silver': 2,
        'gold': 3,
        'platinum': 4,
        'diamond': 5,
        'mythic': 6,
    }
    drafts = (dl.drafts
        .filter(pl.col('draft_time').str.to_datetime().is_between(
            first_draft, last_draft
        ))
        .sort(['pack_number', 'pick_number'])
        .group_by('draft_id').agg(
            pl.col('pick_id'),
            pl.col('pack')
              .alias('packs_ids'),
            pl.col('event_match_wins').max(),
            pl.col('event_match_losses').max(),
            pl.col('rank').mode().first().replace_strict(rank_dict),
            pl.col('user_game_win_rate_bucket').mean()
              .alias('player_win_rate'),
            pl.col('user_n_games_bucket').max()
              .alias('weight'),
        )
        .with_columns(
            pl.col('pick_id').list.to_array(n_picks),
            pl.col('packs_ids').list.to_array(n_picks),
        )
    )
    n_incomplete = drafts.filter(
        (pl.col('event_match_wins') < 7) & (pl.col('event_match_losses') < 3)
    ).select(pl.len()).collect().item()
    assert(n_incomplete == 0)
    if first_n > 0:
        drafts = drafts.head(n=first_n)
    drafts = drafts.collect()
    if verbose > 0:
        select_drafts = len(drafts)
        all_drafts = dl.drafts.select(pl.len()).collect().item() // n_picks
        print(
            f'{CSI}34m{select_drafts} drafts{CSI}0m out of '
            f'{CSI}31m{all_drafts}{CSI}0m '
            f'({100 * select_drafts // all_drafts}%)'
        )
    drafts = Drafts(
        set_id=jnp.full((len(drafts),), set_id, dtype=jnp.int16),
        packs=jnp.array(np.pad(
            drafts['packs_ids'].to_numpy(),
            ((0, 0), (0, 45 - n_picks), (0, 0)),
            constant_values=pad_id
        )),
        picks=jnp.array(np.pad(
            drafts['pick_id'].to_numpy(),
            ((0, 0), (0, 45 - n_picks)),
            constant_values=pad_id
        )),
        game_outcome=jnp.stack((
            drafts['event_match_wins'].to_numpy(),
            drafts['event_match_losses'].to_numpy()
        ), axis=1),
        rank=jnp.array(drafts['rank'].to_numpy(), dtype=jnp.int16),
        player_wr=jnp.array(drafts['player_win_rate'].to_numpy()),
        weight=jnp.array(drafts['weight'].to_numpy()),
    )
    pickle.dump(drafts, open(f'cache/{cache}.pickle', 'wb'))
    return drafts

def collate_cards(dt: DL17Lands) -> Cards:
    card_ids = jnp.array(dt.cards['id'].to_numpy())
    return Cards(
        card_id=card_ids,
        textual_features=jnp.zeros((card_ids.shape[0], 0), dtype=jnp.float32),
        numeric_features=jnp.transpose(jnp.array([
            dt.cards[category]
            for category in [
                'opening_hand', 'drawn', 'tutored', 'deck', 'sideboard', 'GIH'
            ]
        ]))
    )

def add_nlp_features(
    oracle: Any, nlp_processor: NLPProcessor, prompt: Optional[str]=None
) -> Float[Array, 'n_cards d_t']:
    return jnp.array(nlp_processor(
        oracle, prompt=prompt
    ))

def make_reverse_dict(l: Int[Array, "n"], offset=0) -> Int[Array, "m"]:
    assert(l.min() > 0)
    d = np.full(l.max() + 1, -1, dtype=jnp.int32)
    d[0] = 0
    d[l] = offset + np.arange(l.shape[0])
    return jnp.array(d)

def make_graph_knn(
    sims: Float[Array, "n n"], density: float=0.2
) -> Int[Array, "2 m"]:
    n = sims.shape[0]
    k = int((n-1) * density)
    U = []
    V = []
    for u in range(n):
        U.extend([u] * (k+1))
        V.extend(np.argsort(sims[u])[-k-1:])
    return jnp.stack([jnp.array(U), jnp.array(V)])

def make_graph_global(
    sims: Float[Array, "n n"], density: float=0.2
) -> Int[Array, "2 m"]:
    n = sims.shape[0]
    values = sims.flatten()
    values.sort()
    values = values[:-n]
    limit = values[-int(len(values)*density)]
    adj_matrix = sims > limit
    return jnp.argwhere(adj_matrix).T

def make_graph(
    sims: Float[Array, "n n"], density: float=0.1, local: bool=False
) -> Int[Array, "2 m"]:
    if local:
        return make_graph_knn(sims, density)
    return make_graph_global(sims, density)

def make_adjacency(
    graph: Int[Array, "2 m"], n_nodes: int
) -> Bool[Array, "n_nodes n_nodes"]:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=jnp.bool)
    adjacency[graph[0], graph[1]] = True
    return adjacency #type: ignore


class JaxDraftDataset:
    def __init__(
        self,
        dataloaders: List[DL17Lands],
        time_offset: int=0,
        time_window: int=7,
        first_drafts: bool=False,
        pad_id: int=0,
        seed: int=42,
        verbose: int=0
    ):
        self.seed = seed
        self.rng = None

        self.oracles = []
        cards = []
        sets = []
        drafts = []
        offset = 1
        for i, dl in enumerate(dataloaders):
            st = time.time()
            self.oracles.append(dl.cards['oracle'].to_list())
            cards.append(collate_cards(dl))
            id_map = make_reverse_dict(cards[-1].card_id, offset)
            graph = jnp.zeros((2, 0), jnp.int32)
            adjacency = jnp.zeros((0, 0), jnp.int32)
            sets.append(Sets(
                card_ids=id_map[cards[-1].card_id],
                set_size=cards[-1].card_id.shape[0],
                pack_size=dl.pack_size,
                graph=graph,
                adjacency=adjacency
            ))
            cur_drafts = collate_drafts(
                i, dl, time_offset=time_offset, time_window=time_window,
                first_drafts=first_drafts, pad_id=pad_id, verbose=verbose
            )
            cur_drafts = eqx.tree_at(
                lambda d: d.packs,
                cur_drafts,
                id_map[cur_drafts.packs]
            )
            cur_drafts = eqx.tree_at(
                lambda d: d.picks,
                cur_drafts,
                id_map[cur_drafts.picks]
            )
            drafts.append(cur_drafts)
            offset += cards[-1].card_id.shape[0]
            if verbose > 0:
                print(f"Collating {dl.format} took {time.time() - st:.6f}s")
        d_t = cards[-1].textual_features.shape[1] # == 0
        d_n = cards[-1].numeric_features.shape[1]
        cards.insert(0, Cards(
            card_id=jnp.array([0]),
            textual_features=jnp.zeros((1, d_t)),
            numeric_features=jnp.zeros((1, d_n), dtype=jnp.float32)
        ))
        max_set_size = max(s.set_size for s in sets)
        for i in range(len(sets)):
            sets[i] = eqx.tree_at(
                lambda s: s.card_ids,
                sets[i],
                jnp.pad(
                    sets[i].card_ids,
                    ((0, max_set_size - sets[i].set_size),),
                    constant_values=pad_id
                )
            )
        self.cards = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *cards
        )
        self.sets = jax.tree.map(
            lambda *x: jnp.stack(x),
            *sets
        )
        self.drafts = jax.tree.map(
            lambda *x: jnp.concatenate(x, axis=0),
            *drafts
        )
        if not (self.drafts.packs.min() == 0 and self.drafts.picks.min() == 0):
            print(f'{CSI}33mWarning: {CSI}34mLowest pick id is greater than 0 (Min packs ids: {CSI}32m{self.drafts.packs.min()}{CSI}34m; Min pick id: {CSI}32m{self.drafts.picks.min()}{CSI}34m).{CSI}0m')
            assert self.drafts.packs.min() >= 0 and self.drafts.picks.min() >= 0

        self.card_bytes = sum(
            jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.cards))
        )
        self.set_bytes = sum(
            jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.sets))
        )
        self.draft_bytes = sum(
            jax.tree.leaves(jax.tree.map(lambda d: d.nbytes, self.drafts))
        )
        if verbose > 0:
            format_fn = humanfriendly.format_size
            print(f'Cards use:  {format_fn(self.card_bytes):>9}')
            print(f'Sets use:   {format_fn(self.set_bytes):>9}')
            print(f'Drafts use: {format_fn(self.draft_bytes):>9}')

        self.step_fn = None
        self.trainer = None
        self.compiled = None

    def reset(self):
        self.trainer = None
        self.step_fn = None
        self.scan_fn = None
        self.compiled = None

    def process_data(
        self,
        encoder: NLPProcessor,
        graph_density: float=0.1,
        graph_type: Literal['knn', 'global']='knn',
        verbose: int=0
    ):
        st = time.time()
        text_feats = []
        graphs = []
        adjacencies = []
        for i_set, oracle in enumerate(self.oracles):
            text_feats.append(add_nlp_features(
                oracle, encoder,
                "task: Magic the Gathering card selection in a draft | card: "
            ))
            if isinstance(encoder, Gemma):
                h = torch.from_numpy(np.array(text_feats[-1])) #type: ignore
                graph = make_graph(
                    encoder.model.similarity(h, h).numpy(), #type: ignore
                    density=graph_density,
                    local=graph_type=='knn'
                )
                adjacency = make_adjacency(graph, text_feats[-1].shape[0])
            else:
                raise NotImplementedError
            graphs.append(graph)
            adjacencies.append(adjacency)
        d_t = text_feats[-1].shape[1]
        text_feats.insert(0, jnp.zeros((1, d_t), dtype=jnp.float32))
        self.cards = eqx.tree_at(
            lambda c: c.textual_features,
            self.cards,
            jnp.concatenate(text_feats, axis=0)
        )
        max_set_size = self.sets.set_size.max()
        max_graph_size = max(graph.shape[1] for graph in graphs)
        for i_set in range(len(self.oracles)):
            graphs[i_set] = jnp.pad(
                graphs[i_set],
                ((0, 0), (0, max_graph_size - graphs[i_set].shape[1])),
                constant_values=max_set_size
            )
            adjacencies[i_set] = jnp.pad(
                adjacencies[i_set],
                (
                    (0, max_set_size - self.sets[i_set].set_size),
                    (0, max_set_size - self.sets[i_set].set_size)
                ),
                constant_values=False
            )
        self.sets = eqx.tree_at(
            lambda s: s.graph,
            self.sets,
            jnp.stack(graphs)
        )
        self.sets = eqx.tree_at(
            lambda s: s.adjacency,
            self.sets,
            jnp.stack(adjacencies)
        )

        if verbose > 0:
            print(f"Post-processing cards took {time.time() - st:.6f}s")

    def n_steps(self, batch_size) -> int:
        return self.drafts.picks.shape[0] // batch_size

    def shard_data(self, trainer: Optional[Trainer]=None):
        if trainer is not None:
            self.trainer = trainer
        if self.trainer is not None:
            self.cards, self.sets = self.trainer.shard_model(
                self.cards, self.sets
            )
            # self.drafts = trainer.shard_data(self.drafts)

    def set_step_function(self,
        static: DraftWRPredictor,
        step_fn: Callable[
            [
                DraftWRPredictor, eqx.nn.State, Any, Any,
                Cards, Sets, Drafts, PRNGKeyArray
            ],
            Tuple[
                DraftWRPredictor, eqx.nn.State, Any, Float[Array, "..."],
                Float[Array, "bs 45"], Float[Array, "bs 2"],
                Bool[Array, "bs 45"]
            ]
        ]
    ):
        def foo(
            carry: Tuple[
                DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray, Cards,
                Sets
            ],
            batch: Drafts,
            static: DraftWRPredictor,
        ) -> Tuple[
            Tuple[
                DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray, Cards,
                Sets
            ],
            Tuple[
                Float[Array, "..."], Float[Array, "bs 45"],
                Float[Array, "bs 2"], Bool[Array, "bs 45"]
            ]
        ]:
            (
                params, state, opt_state, lr_transform_state, key, cards, sets
            ) = carry
            model = eqx.combine(params, static)
            key, subkey = jax.random.split(key)
            model, state, opt_state, output, pred, true, mask = step_fn(
                model, state, opt_state, lr_transform_state,
                cards, sets, batch, subkey
            )
            params, _ = eqx.partition(model, eqx.is_array)
            return (
                params, state, opt_state, lr_transform_state, key, cards, sets
            ), (output, pred, true, mask)
        self.step_fn = jax.jit(
            partial(
                foo, static=static,
            ),
            donate_argnums=(0, 1)
        )
        self.scan_fn = partial(jax.lax.scan, f=self.step_fn)
        self.compiled = None
        self.rng = np.random.default_rng(self.seed)

    def prepare_indices(self, batch_size: int, shuffle: bool=False):
        assert self.rng is not None
        n_samples = self.drafts.picks.shape[0]
        if 2 * n_samples < batch_size:
            raise ValueError(
                f"Batch size is too large for the dataset (n={n_samples})"
            )
        indices = np.arange(n_samples)
        if n_samples % batch_size != 0:
            duplicate = self.rng.choice(
                jnp.arange(n_samples),
                size=batch_size - (n_samples % batch_size),
                replace=False
            )
            indices = np.concatenate((indices, duplicate), axis=0)
        if shuffle:
            self.rng.shuffle(indices)
        indices = jnp.array(indices).reshape((-1, batch_size))
        return indices

    def precompile(
        self,
        params: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any, #optax.OptState,
        lr_transform_state: Any,
        batch_size: int,
        key: PRNGKeyArray,
        verbose: int=0,
        verbose_name: str='unnamed'
    ):
        if self.compiled == batch_size:
            return
        st = time.time()
        indices = self.prepare_indices(batch_size)
        drafts = self.drafts[indices]
        if self.trainer is not None:
            drafts = self.trainer.shard_data(drafts)
        self.scan_fn = jax.jit(self.scan_fn).trace( #type: ignore
            init=(
                params, state, opt_state, lr_transform_state, key, self.cards,
                self.sets
            ),
            xs=drafts
        ).lower().compile()
        self.compiled = batch_size
        if verbose > 0:
            print(f'Precompiled {verbose_name} loop in {time.time() - st:.6f}s')

    @jaxtyped(typechecker=typechecker)
    def run_batches(
        self,
        params: DraftWRPredictor,
        state: eqx.nn.State,
        opt_state: Any, #optax.OptState,
        lr_transform_state: Any,
        batch_size: int,
        shuffle: bool,
        key: PRNGKeyArray
    ) -> Tuple[
        DraftWRPredictor, eqx.nn.State, Any, Any, PRNGKeyArray,
        Float[Array, "n_batches ..."], Float[Array, "n_batches bs 45"],
        Int[Array, "n_batches bs 2"], Bool[Array, "n_batches bs 45"]
    ]:
        if self.step_fn is None:
            raise ValueError("step_fn is not set")
        if self.compiled is not None and self.compiled != batch_size:
            raise ValueError(
                f"Step function was compiled for batch size {self.compiled}, "
                f"but {batch_size} was used"
            )
        assert self.scan_fn is not None

        indices = self.prepare_indices(batch_size, shuffle)
        drafts = self.drafts[indices]
        if self.trainer is not None:
            drafts = self.trainer.shard_data(drafts)

        (
            (params, state, opt_state, lr_transform_state, key, _, _),
            (outputs, pred, true, mask)
        ) = self.scan_fn(
            init=(
                params, state, opt_state, lr_transform_state, key, self.cards,
                self.sets
            ),
            xs=drafts
        )
        self.compiled = batch_size
        return (
            params, state, opt_state, lr_transform_state, key, outputs, pred,
            true, mask
        )

    def to_serializable(self):
        def to_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, jnp.ndarray):
                return np.array(x)
            return x

        serial = {
            'scalar': {
                'card_bytes': self.card_bytes,
                'set_bytes': self.set_bytes,
                'draft_bytes': self.draft_bytes,
                'seed': self.seed,
                'oracles': self.oracles,
            },
            'eqx_data': {
                'cards': jax.tree.map(to_host, self.cards),
                'drafts': jax.tree.map(to_host, self.drafts),
                'sets': jax.tree.map(to_host, self.sets),
            },
        }
        return serial

    @classmethod
    def from_serialized(cls, serial):
        """Create an instance quickly from serial (dict of numpy arrays)."""
        obj = object.__new__(cls)

        def from_host(x):
            if hasattr(x, "device_buffer") or isinstance(x, np.ndarray):
                return jnp.array(x)
            return x

        for k, v in serial['scalar'].items():
            setattr(obj, k, v)
        obj.cards = jax.tree.map(from_host, serial['eqx_data']['cards'])
        obj.drafts = jax.tree.map(from_host, serial['eqx_data']['drafts'])
        obj.sets = jax.tree.map(from_host, serial['eqx_data']['sets'])
        obj.step_fn = None
        obj.trainer = None
        obj.compiled = None
        obj.rng = None
        return obj

def make_dataset_from_args(args: Dict[str, Any], verbose: int=0) -> Tuple[
    JaxDraftDataset, JaxDraftDataset, JaxDraftDataset
]:
    return make_datasets(
        train_set=args['train_set'],
        val_set=args['val_set'],
        test_set=args['test_set'],
        temporal_split=args['temporal_split'],
        time_window=args['time_window'],
        verbose=verbose
    )

def make_datasets(
    train_set: List[str],
    val_set: List[str],
    test_set: List[str],
    temporal_split: bool=False,
    time_window: int=7,
    seed: int=42,
    verbose: int=0,
    cache: bool=True
) -> Tuple[
    JaxDraftDataset, JaxDraftDataset, JaxDraftDataset
]:
    args = locals()
    hs = None
    if cache:
        hs = hashlib.sha256(
            json.dumps(args, sort_keys=True).encode('utf-8')
        ).hexdigest()
        if os.path.exists(f'cache/datasets/{hs}.pkl'):
            with open(f'cache/datasets/{hs}.pkl', 'rb') as f:
                _args, (data_train, data_val, data_test) = pickle.load(f)
                if args != _args:
                    raise ValueError('Cached arguments do not match')
                return (
                    JaxDraftDataset.from_serialized(data_train),
                    JaxDraftDataset.from_serialized(data_val),
                    JaxDraftDataset.from_serialized(data_test)
                )

    dataloaders_train = [
        DL17Lands(ext, verbose=verbose) for ext in train_set
    ]
    if temporal_split:
        dataloaders_test = dataloaders_train
        dataloaders_val = dataloaders_train
    else:
        dataloaders_test = [
            DL17Lands(ext, verbose=verbose) for ext in test_set
        ]
        if val_set == 'train':
            dataloaders_val = dataloaders_train
        else:
            dataloaders_val = [
                DL17Lands(ext, verbose=verbose) for ext in val_set
            ]

    time_offset = 2*time_window if temporal_split else 0
    if not temporal_split and val_set == 'train':
        time_offset = time_window
    data_train = JaxDraftDataset(
        dataloaders_train,
        time_offset=time_offset,
        time_window=time_window,
        seed=seed,
        verbose=verbose
    )
    data_val = JaxDraftDataset(
        dataloaders_val,
        time_offset=time_window if temporal_split else 0,
        time_window=time_window,
        seed=seed,
        verbose=verbose
    )
    data_test = JaxDraftDataset(
        dataloaders_test,
        time_window=time_window,
        seed=seed,
        verbose=verbose
    )
    if cache:
        assert hs is not None
        os.makedirs('cache/datasets', exist_ok=True)
        with open(f'cache/datasets/{hs}.pkl', 'wb') as f:
            pickle.dump((args, (
                data_train.to_serializable(),
                data_val.to_serializable(),
                data_test.to_serializable()
            )), f)
    return data_train, data_val, data_test

@cache
def fine_tuning_dataset(
    sets: List[str],
    n_days: int,
    seed: int=42,
    verbose: int=0,
    val_split: float=0.2,
    cache: bool=True
) -> Tuple[JaxDraftDataset, JaxDraftDataset]:
    args = {
        'sets': sets,
        'time_window': n_days,
        'first_drafts': True,
        'seed': seed,
    }
    hs = None
    if cache:
        hs = hashlib.sha256(
            json.dumps(args, sort_keys=True).encode('utf-8')
        ).hexdigest()
        if os.path.exists(f'cache/datasets/ft-{hs}.pkl'):
            with open(f'cache/datasets/ft-{hs}.pkl', 'rb') as f:
                try:
                    _args, data, data_val = pickle.load(f)
                except Exception as e:
                    print(e, hs)
                    import sys
                    sys.exit()
                if args != _args:
                    raise ValueError('Cached arguments do not match')
                return (
                    JaxDraftDataset.from_serialized(data),
                    JaxDraftDataset.from_serialized(data_val)
                )

    dataloaders = [
        DL17Lands(ext, verbose=verbose) for ext in sets
    ]

    days_val = max(1, int(n_days * val_split))
    days_test = n_days - days_val
    if days_test <= 0:
        raise ValueError(f'Not enough days: Val[{days_val}] vs Test[{days_test}]')
    data_val = JaxDraftDataset(
        dataloaders,
        time_offset=days_test,
        time_window=days_val,
        first_drafts=True,
        seed=seed,
        verbose=verbose
    )
    data = JaxDraftDataset(
        dataloaders,
        time_window=days_test,
        first_drafts=True,
        seed=seed,
        verbose=verbose
    )
    if cache:
        assert hs is not None
        os.makedirs('cache/datasets', exist_ok=True)
        with open(f'cache/datasets/ft-{hs}.pkl', 'wb') as f:
            pickle.dump(
                (args, data.to_serializable(), data_val.to_serializable()),
                f
            )
    return data, data_val


def train_test_split(df, test_size=0.2, seed=0):
    return df.with_columns(
        pl.int_range(pl.len(), dtype=pl.UInt32)
        .shuffle(seed=seed)
        .gt(pl.len() * test_size)
        .alias('split')
    ).partition_by('split', include_key=False)

def test_set(ext):
    print("#" * 50)
    print(f"Test {ext}")
    print("=" * 10)
    dataloader= DL17Lands(ext, verbose=True)
    print(dataloader.all_cards.shape)
    print(dataloader.cards.shape)
    with pl.Config(tbl_cols=-1):
        print(dataloader.cards.head())
        print(dataloader.drafts.collect().shape)
        print(dataloader.drafts.head().collect())
    print(dataloader.cards.estimated_size())
    print(dataloader.drafts.collect().estimated_size())
    picks_counts = dataloader.drafts.group_by('draft_id').len().select(
        pl.col('len').alias('n_picks')
    ).group_by('n_picks').len()
    max_picks = picks_counts.select(pl.max('n_picks')).collect()['n_picks'][0]
    wl_stats = dataloader.drafts.group_by('draft_id').agg(
        pl.col('event_match_wins').mean(), pl.col('event_match_losses').mean()
    ).select(
        pl.col('event_match_wins').sum(), pl.col('event_match_losses').sum()
    ).collect()
    wins = wl_stats['event_match_wins'][0]
    losses = wl_stats['event_match_losses'][0]
    return (
        ext,
        dataloader.cards.shape[0],
        picks_counts.filter(pl.col('n_picks') == max_picks).collect()['len'][0],
        dataloader.pack_size,
        wins / (wins + losses)
    )

def main():
    data = []
    for exp in [
        'NEO','ONE','MOM','WOE','LCI','MKM','OTJ','BLB','DSK','FDN','DFT','TDM',
        'FIN','EOE','SOS'
    ]:
        data.append(test_set(exp))

    print("#" * 50)
    for exp, n_cards, n_drafts, pack_size, wr in data:
        print(f"{exp} & 00-00-2021 & {n_cards} & {pack_size} & {n_drafts} & {wr:.4f} \\")

if __name__ == "__main__":
    main()
