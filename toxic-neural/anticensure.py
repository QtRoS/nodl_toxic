import argparse
import os
import re

import pandas as pd
from tqdm import tqdm, tqdm_pandas
tqdm.pandas(tqdm())

mapping = {
    'a***es': 'asshole',
    'a**hoe': 'asshole',
    'a**hole': 'asshole',
    'a*s': 'ass',
    'ars*hole': 'asshole',
    'as**ole': 'asshole',
    'as*fu*king': 'fuck',
    'as*hol*s': 'asshole',
    'as*hole': 'asshole',
    'ass*hole': 'asshole',
    'assh*le': 'asshole',
    "assh*le's": 'asshole',
    'assh*ole': 'asshole',
    'b******d': 'bitard',
    'b***er': 'bitard',
    'b***h': 'bitch',
    'b**ch': 'bitch',
    'b**ching': 'bitch',
    'b**tard': 'bitard',
    'b*ch': 'bitch',
    'b*ll': 'balls',
    'b*lls': 'balls',
    'b*stard': 'bastard',
    'b*tch': 'bitch',
    'ba*tard': 'bitard',
    'bi*ch': 'bitch',
    'bl**dy': 'bloody',
    'bulls**t': 'bullshit',
    'bulls*it': 'bullshit',
    'bullsh*t': 'bullshit',
    'c*******ing': 'cocksucking',
    'c***s': 'cunts',
    'c**t': 'cunt',
    'c*ck': 'cock',
    'c*nt': 'cunt',
    'c*ntfaced': 'cunt',
    'c*nts': 'cunts',
    'clusterf*ck': 'fuck',
    'co**sucker': 'cocksucker',
    'cocksu*_cking': 'cocksucking',
    'cr*p': 'crap',
    'cr*ppy': 'crap',
    'd***s': 'dicks',
    'd**b**s': '',
    'd**n': '',
    'd*ck': 'dick',
    'd*irty': 'dirty',
    'd*mb': 'dumb',
    'd*mn': 'damn',
    'di*k': 'dick',
    'di*ks': 'dicks',
    'f*(&n': 'fuck',
    'f*(x': 'fuck',
    'f**%!ng': 'fuck',
    'f*****g': 'fuck',
    'f****er': 'fuck',
    'f****r': 'fuck',
    'f***er': 'fuck',
    'f***ers': 'fuck',
    'f***in': 'fuck',
    'f***ing': 'fuck',
    'f***king': 'fuck',
    'f**ck': 'fuck',
    'f**cking': 'fuck',
    'f**in': 'fuck',
    'f**k': 'fuck',
    "f**k's": 'fuck',
    'f**k**g': 'fuck',
    'f**kin': 'fuck',
    'f**king': 'fuck',
    'f**kker': 'fuck',
    'f**ks': 'fuck',
    'f*ck': 'fuck',
    'f*ck*ng': 'fuck',
    'f*ck.and': 'fuck',
    'f*cked': 'fuck',
    'f*cker': 'fuck',
    'f*ckhead': 'fuck',
    'f*ckin': 'fuck',
    'f*cking': 'fuck',
    'f*ggots': 'faggot',
    'f*ing': 'fuck',
    'f*k': 'fuck',
    'f*u*c*k': 'fuck',
    'f*ucked': 'fuck',
    'f*ucking': 'fuck',
    'f_uc*ck': 'fuck',
    'fagg*t': 'faggot',
    'fc*k': 'fuck',
    'fu*er': 'fuck',
    'fu*k': 'fuck',
    'fu*ker': 'fuck',
    'fu*ker.sikh': 'fuck',
    'fu*khead': 'fuck',
    'fu*king': 'fuck',
    'fu*kwit': 'fuck',
    'fuc*ers': 'fuck',
    'fuc*in': 'fuck',
    'fuc*ing': 'fuck',
    'fuckfu*ker': 'fuck',
    'h*ll': 'hell',
    'h*tler': 'hitler',
    'hon*ey': 'honey',
    'id*ot': 'idiot',
    'id*t': 'idiot',
    'j*sus': 'jesus',
    'm*a*s*h': 'mash',
    'moderf***n': 'motherfucker',
    'mot*herfuc*ker': 'motherfucker',
    'moth*rfucker': 'motherfucker',
    'mothaf*cka': 'motherfucker',
    'mother****er': 'motherfucker',
    'motherf***in': 'motherfucker',
    'motherf**ker': 'motherfucker',
    'motherf**king': 'motherfucker',
    'motherf*cker': 'motherfucker',
    'motherf*cking': 'motherfucker',
    'motherfu*_ccker': 'motherfucker',
    'n*****r': 'nigger',
    'n****r': 'nigger',
    'n**ger': 'nigger',
    'n*gga': 'nigger',
    'n*zi': 'nazi',
    'nig*er': 'nigger',
    'nig*ger': 'nigger',
    'p*opd*ck': 'dick',
    'r*t*rd': 'retard',
    're***d': 'retard',
    'ret*rds': 'retard',
    's*#t': 'shit',
    's*****”': 'shit',
    's**t': 'shit',
    's*cks': 'sucks',
    's*ck': 'suck',
    's*it': 'shit',
    'sh**t': 'shit',
    'sh*t': 'shit',
    'sh*thole': 'asshole',
    'sh*ting': 'shit',
    'sh*tty': 'shit',
    'starfu*k': 'fuck',
    'su**s': 'sucks',
    't*rd': 'tard',
    't*ts': 'tits',
    'un*sexyness': 'unisex',
    'v*rgins': 'virgins',
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_in')
    parser.add_argument('test_in')
    parser.add_argument('train_out')
    parser.add_argument('test_out')
    return parser.parse_args()


# Fast and case-sensitive
# def transform_line(line):
#     for k,v in mapping.items():
#         line = line.replace(k, v)
#     return line

mapping_re = [ (re.compile(re.escape(k), re.IGNORECASE), v) for k, v in mapping.items()]
def transform_line(line):
    for r, v in mapping_re:
        line = r.sub(v, line)
    return line
# def transform_line(line):
#     for k,v in mapping.items():
#         insensitive_re = re.compile(re.escape(k), re.IGNORECASE)
#         line = insensitive_re.sub(v, line)
#     return line

def main():
    args = get_args()
    print('ARGS:', args)
    
    print('Transforming train...')
    df = pd.read_csv(args.train_in)
    df['comment_text'] = df['comment_text'].progress_apply(transform_line)
    df.to_csv(args.train_out, index=False)

    print('Transforming test...')
    df = pd.read_csv(args.test_in)
    df['comment_text'] = df['comment_text'].progress_apply(transform_line)
    df.to_csv(args.test_out, index=False)

if __name__ == '__main__':
    main()
