__author__ = 'Tushar'

"""
Preprocessing script for SICK data.

"""

import os
import glob
cmd1 = ('cd ../lib')
def make_dirs(dirs):
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

def constituency_parse(filepath, cp='', tokenize=True):
    dirpath = os.path.dirname(filepath)
    print('\nDirpath ' + dirpath)
    filepre = os.path.splitext(os.path.basename(filepath))[0]
    print('\nfilepre ' + filepre)
    tokpath = os.path.join(dirpath, filepre + '.toks')
    print('\ntokpath ' + tokpath)
    parentpath = os.path.join(dirpath, filepre + '.cparents')
    print('\nparentpath ' + parentpath)
    tokenize_flag = '-tokenize - ' if tokenize else ''
    print('\ntok_flag ' + tokenize_flag)
    cmd = ('java ConstituencyParse -deps - %s-tokpath %s -parentpath %s < %s'
        %(tokenize_flag, tokpath, parentpath, filepath))
    print('\nDcmd:::::::\n\n\n ' + cmd)
    os.chdir('C:\\treelstm-master\\scripts')
    os.system(cmd)

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

def split(filepath, dst_dir):
    with open(filepath) as datafile, \
         open(os.path.join(dst_dir, 'a.txt'), 'w') as afile:
            datafile.readline()
            for line in datafile:
                i, a, b, sim, ent = line.strip().split('\t')
                #idfile.write(i + '\n')
                afile.write(a + '\n')

def parse(dirpath, cp=''):
    constituency_parse(os.path.join(dirpath, 'a.txt'), cp=cp, tokenize=True)

if __name__ == '__main__':
    print('=' * 80)
    print('Preprocessing SICK dataset')
    print('=' * 80)

    base_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    sick_dir = os.path.join(data_dir, 'sick')
    lib_dir = os.path.join(base_dir, 'lib')
    train_dir = os.path.join(sick_dir, 'train')
    dev_dir = os.path.join(sick_dir, 'dev')
    test_dir = os.path.join(sick_dir, 'test')
    make_dirs([train_dir, dev_dir, test_dir])

    classpath = ':'.join([
        lib_dir,
        os.path.join(lib_dir, 'stanford-parser/stanford-parser.jar'),
        os.path.join(lib_dir, 'stanford-parser/stanford-parser-3.5.1-models.jar')])
    # split into separate files
    split(os.path.join(sick_dir, 'SICK_train.txt'), train_dir)
#    split(os.path.join(sick_dir, 'SICK_trial.txt'), dev_dir)
#    split(os.path.join(sick_dir, 'SICK_test_annotated.txt'), test_dir)

    parse(train_dir, cp=classpath)
#    parse(dev_dir, cp=classpath)
#    parse(test_dir, cp=classpath)

    # get vocabulary
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab.txt'))
    build_vocab(
        glob.glob(os.path.join(sick_dir, '*/*.toks')),
        os.path.join(sick_dir, 'vocab-cased.txt'),
        lowercase=False)
