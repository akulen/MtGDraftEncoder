from functools import cache, partial
import json
import os
import re
import requests
import polars as pl
from typing import cast, List

from replay_dtypes import get_dtypes

os.makedirs('data', exist_ok=True)
CSI = "\x1b[" #]

HEADERS = {
    'User-Agent': 'MtGDraftEncoder/1.0',
    'Accept': 'application/json;q=0.9,*/*;q=0.8'
}

@cache
def get_all_cards():
    filename = 'data/cards_mtga.csv'
    if not os.path.isfile(filename):
        url = (
            'https://17lands-public.s3.amazonaws.com/'
            'analysis_data/cards/cards.csv'
        )
        response = requests.get(url)
        with open('data/cards_mtga.csv', 'wb') as file:
            file.write(response.content)
    dtypes = get_dtypes(filename=filename)
    return pl.read_csv(filename, schema=dtypes)

SCRYFALL_BASE_URL = """
    https://api.scryfall.com/cards/named?exact={}&format=image&version=normal
"""
card_image_cache = None
def get_card_image(card_name):
    global card_image_cache
    if card_image_cache is None:
        if not os.path.isfile('cache/card_images.json'):
            card_image_cache = {}
        else:
            with open('cache/card_images.json', 'r') as file:
                card_image_cache = json.load(file)
    if card_name not in card_image_cache:
        try:
            response = requests.get(
                SCRYFALL_BASE_URL.format(card_name), headers=HEADERS
            )
            response.raise_for_status()
            card_image_cache[card_name] = response.url
            with open('cache/card_images.json', 'w') as file:
                json.dump(card_image_cache, file)
        except Exception:
            return "https://via.placeholder.com/140x200?text=Image+Error"
    return card_image_cache[card_name]

@cache
def get_oracle():
    if os.path.isfile('data/cards_scryfall.parquet'):
        return pl.read_parquet('data/cards_scryfall.parquet')
    if not os.path.isfile('data/cards_scryfall.json'):
        url = 'https://api.scryfall.com/bulk-data/oracle-cards'
        response = requests.get(url, headers=HEADERS)
        data = response.json()
        response = requests.get(data['download_uri'], headers=HEADERS)
        scryfall_oracle = response.json()
        with open('data/cards_scryfall.json', 'w') as file:
            json.dump(scryfall_oracle, file)
    oracle_data = pl.read_json(
        'data/cards_scryfall.json',
        infer_schema_length=None
    )
    # Drop all sets with cards that can have duplicate names
    oracle_data = oracle_data.filter(~pl.col('set').is_in([
        'ugl', 'ust', 'unf', 'cmb2'
    ]))
    # Drop all tokens
    oracle_data = oracle_data.filter(~pl.col('set_type').is_in([
        'token', 'memorabilia'
    ]))
    oracle_data.write_parquet(
        'data/cards_scryfall.parquet',
        use_pyarrow=True,
    )
    return oracle_data

def full_oracle_polars_V1(df):
    front_face = pl.col('card_faces').list.get(0).struct
    back_face = pl.col('card_faces').list.get(1).struct
    df = df.with_columns(
        faces_oracle=(
            pl.when(pl.col('oracle_text').is_null())
            .then(pl.concat_str([
                front_face.field('oracle_text'),
                back_face.field('oracle_text'),
            ], separator=' // '))
            .otherwise(pl.col('oracle_text'))
        ),
        faces_cost=(
            pl.when(pl.col('mana_cost').is_null())
            .then(pl.concat_str([
                front_face.field('mana_cost'),
                back_face.field('mana_cost'),
            ], separator=' // '))
            .otherwise(pl.col('mana_cost'))
        ),
        power_box=(
            pl.when(~pl.col('power').is_null())
            .then(pl.concat_str([
                pl.col('power'),
                pl.lit('/'),
                pl.col('toughness'),
            ]))
            .when(~pl.col('loyalty').is_null())
            .then(pl.col('loyalty'))
            .when(~pl.col('card_faces').is_null())
            .then(pl.concat_str([
                pl.when(~front_face.field('power').is_null())
                .then(pl.concat_str([
                    front_face.field('power'),
                    pl.lit('/'),
                    front_face.field('toughness'),
                ]))
                .when(~front_face.field('loyalty').is_null())
                .then(front_face.field('loyalty'))
                .otherwise(pl.lit('')),
                pl.when(~back_face.field('power').is_null())
                .then(pl.concat_str([
                    back_face.field('power'),
                    pl.lit('/'),
                    back_face.field('toughness'),
                ]))
                .when(~back_face.field('loyalty').is_null())
                .then(back_face.field('loyalty'))
                .otherwise(pl.lit('')),
            ], separator=' // '))
            .otherwise(pl.lit(''))
        ),
    )
    df = df.with_columns(oracle=(
          '<name> '
        + pl.col('name')
        + '\n<cost> '
        + pl.col('faces_cost')
        + '\n<type> '
        + pl.col('type_line')
        + '\n<oracle> '
        + pl.col('faces_oracle').str.replace_all('\n', ' <nl> ')
        + '\n<pwl> '
        + pl.col('power_box')
    ))
    return df

def get_mana_value(mana_cost):
    return pl.when(mana_cost.str.contains('//')).then(pl.lit('-1')).otherwise(
        mana_cost
        .str.replace_all(r'\{C\}', '+1')
        .str.replace_all(r'\{W\}', '+1')
        .str.replace_all(r'\{U\}', '+1')
        .str.replace_all(r'\{B\}', '+1')
        .str.replace_all(r'\{R\}', '+1')
        .str.replace_all(r'\{G\}', '+1')
        .str.replace_all(r'\{C/W\}', '+1')
        .str.replace_all(r'\{C/U\}', '+1')
        .str.replace_all(r'\{C/B\}', '+1')
        .str.replace_all(r'\{C/R\}', '+1')
        .str.replace_all(r'\{C/G\}', '+1')
        .str.replace_all(r'\{2/W\}', '+2')
        .str.replace_all(r'\{2/U\}', '+2')
        .str.replace_all(r'\{2/B\}', '+2')
        .str.replace_all(r'\{2/R\}', '+2')
        .str.replace_all(r'\{2/G\}', '+2')
        .str.replace_all(r'\{W/P\}', '+1')
        .str.replace_all(r'\{U/P\}', '+1')
        .str.replace_all(r'\{B/P\}', '+1')
        .str.replace_all(r'\{R/P\}', '+1')
        .str.replace_all(r'\{G/P\}', '+1')
        .str.replace_all(r'\{W/U(/P)?\}', '+1')
        .str.replace_all(r'\{U/B(/P)?\}', '+1')
        .str.replace_all(r'\{B/R(/P)?\}', '+1')
        .str.replace_all(r'\{R/G(/P)?\}', '+1')
        .str.replace_all(r'\{G/W(/P)?\}', '+1')
        .str.replace_all(r'\{W/B(/P)?\}', '+1')
        .str.replace_all(r'\{B/G(/P)?\}', '+1')
        .str.replace_all(r'\{G/U(/P)?\}', '+1')
        .str.replace_all(r'\{U/R(/P)?\}', '+1')
        .str.replace_all(r'\{R/W(/P)?\}', '+1')
        .str.replace_all(r'\{X\}', '+0')
        .str.replace_all(r'\{', '+')
        .str.replace_all(r'\}', '')
        .str.strip_chars('+')
        .str.replace_all(r'^$', '0')
    ).map_elements(lambda x: str(eval(x)), return_dtype=pl.String)

@cache
def get_keywords():
    filename = 'data/keywords.json'
    url = "https://api.scryfall.com/catalog/keyword-abilities"
    if not os.path.isfile(filename):
        keywords = requests.get(url, headers=HEADERS).json()['data']
        with open(filename, 'w') as file:
            json.dump(keywords, file)
    else:
        with open(filename, 'r') as file:
            keywords = json.load(file)
    return keywords

KEYWORDS = get_keywords()
KEYWORDS = '|'.join([re.escape(k) for k in KEYWORDS])
KEYWORDS_REGEX = re.compile(
    r'(?i)\b(?:' + KEYWORDS + r')\b'
).pattern
def make_face_expr(col_fn, rarity=None) -> pl.Expr:
    expr = pl.lit('[NAME] ') + col_fn('name')
    mc = col_fn('mana_cost')
    expr += (pl.when(mc != '')
        .then(
            pl.lit(' | [MC] ') + mc + pl.lit(' | [CMC] ') + get_mana_value(mc)
        ) # TODO: should backside of DFC have mana value of front side?
        .otherwise(pl.lit(''))
    )
    expr += pl.lit(' | [COLOR] ') + pl.when(col_fn('colors').list.len()>0).then(
        col_fn('colors').list.join(', ')
    ).otherwise(pl.lit('C'))
    if rarity is None:
        rarity = col_fn('rarity')
    expr += pl.lit(' | [RARITY] ') + rarity
    type_line = col_fn('type_line').str.split(' — ')
    expr += pl.lit(' | [TYPE] ') + type_line.list.get(0)
    expr += pl.when(type_line.list.len() > 1).then(
        pl.lit(' | [SUBTYPE] ') + type_line.list.get(-1)
    ).otherwise(pl.lit(''))
    expr += pl.when(col_fn('power').is_not_null()).then(
          pl.lit(' | [P/T] ')
        + col_fn('power') + pl.lit('/') + col_fn('toughness')
    ).otherwise(pl.lit(''))
    expr += pl.when(col_fn('loyalty').is_not_null()).then(
        pl.lit(' | [LOYALTY] ') + col_fn('loyalty')
    ).otherwise(pl.lit(''))
    keywords = (
        col_fn('oracle_text')
        .str.replace_all(r' *\([^\)]*\) *', '') # Remove reminder text
        .str.extract_all(KEYWORDS_REGEX)
    )
    expr += pl.when(keywords.list.len() > 0).then(
          pl.lit(' | [KEYWORDS] ')
        + keywords.list.join(', ').str.replace_all(';', ',')
    ).otherwise(pl.lit(''))
    expr += (
          pl.lit(' | [EFFECTS] ')
        + col_fn('oracle_text').str.replace_all('\n', ' <NL> ')
    )
    return expr

def full_oracle_polars(df):
    return df.with_columns(oracle=
        pl.when(~pl.col('oracle_text').is_null())
        .then(make_face_expr(pl.col))
        .otherwise(
              pl.lit('[FACE_1] ')
            + make_face_expr(
                pl.col('card_faces').list.get(0).struct.field,
                pl.col('rarity')
            )
            + pl.lit(' // [FACE_2] ')
            + make_face_expr(
                pl.col('card_faces').list.get(1).struct.field,
                pl.col('rarity')
            )
        )
    )

def make_oracle(df):
    entries = get_oracle().lazy().filter(
        ~pl.col('type_line').str.contains('Token')
    ).with_columns(
        pl.col('name').str.split(' // ').list.get(0).alias('short_name'),
    ).unique(subset='name').join(
        df.lazy(),
        left_on='short_name',
        right_on='name',
        how='right',
        coalesce=False
    ).collect()
    # if len(df_cards.filter(pl.col('oracle').is_null())) > 0:
    #     print(
    #         f'{CSI}31mSome cards were not found. This could be caused by '
    #         'double-faced cards. To fix those, add them to the dbl_face '
    #         'dictionary argument, using the following suggestions:'
    #     )
    #     missing = df_cards.filter(pl.col('oracle').is_null())['name']
    #     for name in missing:
    #         print(f'{CSI}34m[{name}]')
    #         for cand in self.oracle_data.filter(
    #             pl.col('name').str.starts_with(name)
    #         )['name'].to_list():
    #             print(f'\t{CSI}32m{cand}{CSI}0m')
    #     import sys
    #     sys.exit(1)
    return full_oracle_polars(entries)

@cache
def get_draft_df(ext):
    filename = f'data/draft_data_public.{ext}.PremierDraft'
    url = (
        'https://17lands-public.s3.amazonaws.com/analysis_data/'
        f"draft_{filename}.csv.gz"
    )
    if not os.path.isfile(f"{filename}.parquet"):
        if not os.path.isfile(f"{filename}.csv.gz"):
            response = requests.get(url)
            with open(f"{filename}.csv.gz", 'wb') as file:
                file.write(response.content)

        dtypes = get_dtypes(filename=f"{filename}.csv.gz")
        df = pl.scan_csv(f"{filename}.csv.gz", schema=dtypes)
        df.sink_parquet(f"{filename}.parquet")
    else:
        df = pl.scan_parquet(f"{filename}.parquet")
    if 'pick_2' in df.collect_schema().names():
        values = df.select(pl.col('pick_2').unique()).collect()['pick_2'].to_list()
        print(f'{CSI}31mPick 2 is not currently supported ({','.join(map(str, values))}).{CSI}0m')
        df = df.filter(pl.col('pick_2').is_null() | (pl.col('pick_2') == False))
    return df

def fix_split_names(df_cards, ext, verbose=0):
    df = get_draft_df(ext)
    for column in df.collect_schema().names():
        if column[:5] == 'pool_':
            name = column[5:]
            if len(df_cards.filter(pl.col('name')==name)['id']) == 1:
                continue
            if len(df_cards.filter(pl.col('full_name')==name)['id']) != 1:
                print(f'{CSI}31mCard not found:')
                print(f'{CSI}34m[{column}] {CSI}32m{name}{CSI}0m')
                import sys
                sys.exit()
            if verbose > 0:
                print(f'{CSI}34mRenaming{CSI}32m {df_cards.filter(pl.col("full_name")==name)['name'].item()}{CSI}34m to {CSI}32m{name}{CSI}0m')
            df_cards = df_cards.with_columns(
                pl.when(pl.col('full_name') == name)
                .then(pl.lit(name))
                .otherwise(pl.col('name'))
                .alias('name')
            )
    return df_cards

def get_draft_data(df_cards, ext, pack_size=None, pad_id=0) -> pl.LazyFrame:
    filename = f'data/draft_data_public.{ext}.PremierDraft'
    if os.path.isfile(f"{filename}.compact.parquet"):
        return pl.scan_parquet(f"{filename}.compact.parquet")

    df = get_draft_df(ext)

    if pack_size is None:
        from collections import Counter
        distrib = df.group_by('draft_id').len()
        counts = distrib.select(pl.col('len')).collect().to_numpy()
        cnt = Counter()
        for c in counts:
            cnt[c[0]] += 1
        print(f'{ext} is missing pack_size!')
        print('#' * 50)
        print("Distribution of #Picks per draft:")
        for c in sorted(cnt.keys()):
            print(f"    {c:>2} picks: {cnt[c]:>6}")
        print(
            "Maximum pick in a pack:",
            df.select(pl.col("pick_number")).max().collect()['pick_number'][0]+1
        )
        import sys
        sys.exit()

    df = df.filter(
          pl.col('user_game_win_rate_bucket').is_not_null()
        & ( # Remove incomplete drafts
              (pl.col('event_match_wins') == 7)
            | (pl.col('event_match_losses') == 3)
        )
    ).filter(
        pl.col("draft_id").len().over("draft_id") == 3 * pack_size
    )

    cards = {}
    for column in df.collect_schema().names():
        if column[:5] == 'pool_':
            name = column[5:]
            try:
                cards[name] = (
                    df_cards.filter(pl.col('name')==name)['id']
                            .item()
                )
            except ValueError:
                print(f'{CSI}31mCard not found:')
                print(f'{CSI}34m[{column}] {CSI}32m{name}{CSI}0m')
                import sys
                sys.exit()

    df = df.with_columns(
        pick_id=pl.col('pick').replace_strict(cards),
        pool=pl.concat_list([
            pl.lit(card_id).repeat_by(pl.col(f'pool_{card_name}'))
            for card_name, card_id in cards.items()
        ]),
        pack=pl.concat_list([
            pl.lit(card_id).repeat_by(pl.col(f'pack_card_{card_name}'))
            for card_name, card_id in cards.items()
        ] + [
            pl.lit(pad_id).repeat_by(
                15 - pl.sum_horizontal(pl.col('^pack_card_.*$'))
            )
        ]).list.to_array(15)
    ).drop([
        'event_type',
    ] + [
        f'{prefix}_{card_name}'
        for card_name, _ in cards.items()
        for prefix in ['pool', 'pack_card']
    ])
    df = df.collect()
    df.write_parquet(
        f"{filename}.compact.parquet",
        use_pyarrow=True
    )
    return df.lazy()

def card_stats(
    card: str, types: List[str], df_games: pl.DataFrame
) -> float:
    print(df_games.columns)
    col_sum = cast(pl.Series, sum(df_games[f'{tp}_{card}'] for tp in types))

    return (
        (col_sum * df_games['won']).sum()
        /
        max(1, col_sum.sum())
    )

def get_card_stats(
    df_cards, ext, include_ext=False, data_exclude=None
) -> pl.DataFrame:
    if data_exclude is None:
        data_exclude = []
    filename = f'data/game_data_public.{ext}.PremierDraft'
    stat_cols = {
        'opening_hand': ['opening_hand'],
        'drawn': ['drawn'],
        'tutored': ['tutored'],
        'deck': ['deck'],
        'sideboard': ['sideboard'],
        'GIH': ['opening_hand', 'drawn'],
    }
    if not os.path.isfile(f"{filename}.parquet"):
        if not os.path.isfile(f"{filename}.csv.gz"):
            url = (
                'https://17lands-public.s3.amazonaws.com/analysis_data/'
                f"game_{filename}.csv.gz"
            )
            response = requests.get(url)
            with open(f"{filename}.csv.gz", 'wb') as file:
                file.write(response.content)
        dtypes = get_dtypes(filename=f"{filename}.csv.gz")
        df_games = pl.read_csv(f"{filename}.csv.gz", schema=dtypes)

        all_cards = {
            col.split('_')[-1]
            for col in df_games.columns
            if '_' in col
        } - {
            'turns','time','play','rank','number','mulligans','index','bucket',
            'colors','id','type'
        }
        all_cards = pl.DataFrame({'name': list(all_cards)})

        average_wr = df_games['won'].mean()
        def type_exclude(types, data_exclude):
            for tp in types:
                if tp in data_exclude:
                    return True
            return False
        df_stats = all_cards.with_columns(*[
            pl.col('name').map_elements(
                (
                    partial(card_stats, types=types, df_games=df_games)
                    if type_exclude(types, data_exclude)
                    else lambda c: average_wr
                ),
                return_dtype=pl.Float32
            ).alias(f"{ext}_{name}")
            for name, types in stat_cols.items()
        ])
        df_stats = df_stats.with_columns(
            weight=pl.col('name').map_elements(
                lambda x: df_games[f'deck_{x}'].sum(),
                return_dtype=pl.Float32
            ).cast(pl.Int32)
        )
        df_stats.select(
            pl.col('name', *[
                f"{ext}_{name}"
                for name in stat_cols
            ], 'weight')
        ).write_parquet(
            f"{filename}.parquet",
            use_pyarrow=True,
        )
    else:
        df_stats = pl.read_parquet(f"{filename}.parquet")

    c, cs = set(df_cards['name']), set(df_stats['name'])
    if len(c) != len(cs) or len(c - cs) > 0:
        print(f'{CSI}33mWarning: Game card stats probably outdated for {ext}{CSI}0m')
        print(f'{CSI}34m{len(c)} cards requested, but {len(cs)} card stats found.{CSI}0m')
        # print(f'{c-cs} | {cs-c}')
        # raise Exception(f'Game card stats outdated for {ext}')
    df_cards = df_cards.join(df_stats, on='name', how='left')

    if not include_ext:
        df_cards = df_cards.rename({
            f'{ext}_{name}': name
            for name in stat_cols
        })
    return df_cards


def main():
    cards = pl.DataFrame({
        'name': [
            'Questing Druid',
            'Giant Growth',
            'Tamiyo, Inquisitive Student',
            'Sink into Stupor',
            'Valki, God of Lies',
            'Rakshasa\'s Bargain',
            'Angel of Salvation',
            'Ancient Stone Idol',
            'Aquatic Alchemist',
        ],
    })
    cards = cards.with_columns(oracle=make_oracle(cards)) # type: ignore
    with pl.Config(set_fmt_str_lengths=6000):
        print(cards)
    cards = pl.DataFrame({
        'name': [
            'Embercleave',
            'Giant Growth',
            'Doubling Season',
        ],
    })
    cards = get_card_stats(cards, 'FDN')
    with pl.Config(set_fmt_str_lengths=6000):
        print(cards)

if __name__ == "__main__":
    main()
