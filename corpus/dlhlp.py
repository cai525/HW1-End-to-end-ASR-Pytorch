from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset

# Additional (official) text src provided
OFFICIAL_TXT_SRC = ['dlhlpspeech-lm-norm.txt']
# Remove longest N sentence in dlhlpspeech-lm-norm.txt
REMOVE_TOP_N_TXT = 5000000
# Default num. of threads used for loading DlhlpSpeech
READ_FILE_THREADS = 4


def read_text(file:str):
    """Get transcription of target wave file,
       it's somewhat redundant for accessing each txt multiplt times,
       but it works fine with multi-thread

       获得音频的转换文本"""
    file = file.replace('\\','/')
    src_file = file.rsplit('/', 1)[0]+'/bopomo.trans.txt'  # 不包括后缀
    idx = file.split('/')[-1].split('.')[0]

    with open(src_file, 'r',encoding="utf-8") as fp:
        for line in fp:
            if idx == line.split(' ')[0]:
                return line[:-1].split(' ', 1)[1]


class DlhlpDataset(Dataset):
    """
        继承torch的dataset类，用于导入语音-文本数据对
    """

    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        """
        inputs:
            path: 训练集下的路径
            split: 训练集下子文件夹的路径
            tokenizer: 用于将文本进行分词
            bucket_size: 分箱大小，似乎是一种离散化的方法
            ascending: 是否按文本长度下降的方式对文件进行排序

        properties:
            self.file_list: (list<str>) 音频文件（.wav）的路径列表
            self.text: (list<tensor>) 音频文件对应文本(编码后)
        """
        # Setup
        self.path = path
        self.bucket_size = bucket_size

        # List all wave files
        file_list = []
        for s in split:
            # ：检索目录下所有.wav格式的文件（注意这里的join是os.path的join,用来拼接路径）
            split_list = list(Path(join(path, s)).rglob("*.wav"))  # 说明：wav是音频格式
            assert len(split_list) > 0, "No data found @ {}".format(join(path, s))
            file_list += split_list
        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(  # 运用了joblib包来并行以提高效率
            delayed(read_text)(str(f)) for f in file_list)  # 可以只看里面的东西，如果外面的一层封装看不懂的话
        # text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
        text = [tokenizer.encode(txt) for txt in text]

        # Sort dataset by text length
        # file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
        self.file_list, self.text = zip(*[(f_name, txt)  # zip() 与 * 运算符相结合可以用来拆解一个列表（来自python文档）
                                          for f_name, txt in
                                          sorted(zip(file_list, text), reverse=not ascending, key=lambda x: len(x[1]))])

    def __getitem__(self, index):
        if self.bucket_size > 1:
            # Return a bucket
            index = min(len(self.file_list) - self.bucket_size, index)
            return [(f_path, txt) for f_path, txt in
                    zip(self.file_list[index:index + self.bucket_size], self.text[index:index + self.bucket_size])]
        else:
            return self.file_list[index], self.text[index]

    def __len__(self):
        return len(self.file_list)


class DlhlpTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size):
        # Setup
        self.path = path
        self.bucket_size = bucket_size
        self.encode_on_fly = False
        read_txt_src = []

        # List all wave files
        file_list, all_sent = [], []

        for s in split:
            if s in OFFICIAL_TXT_SRC:
                self.encode_on_fly = True
                with open(join(path, s), 'r') as f:
                    all_sent += f.readlines()
            file_list += list(Path(join(path, s)).rglob("*.wav"))
        assert (len(file_list) > 0) or (len(all_sent)
                                        > 0), "No data found @ {}".format(path)

        # Read text
        text = Parallel(n_jobs=READ_FILE_THREADS)(
            delayed(read_text)(str(f)) for f in file_list)
        all_sent.extend(text)
        del text

        # Encode text
        if self.encode_on_fly:
            self.tokenizer = tokenizer
            self.text = all_sent
        else:
            self.text = [tokenizer.encode(txt) for txt in tqdm(all_sent)]
        del all_sent

        # Read file size and sort dataset by file size (Note: feature len. may be different)
        self.text = sorted(self.text, reverse=True, key=lambda x: len(x))
        if self.encode_on_fly:
            del self.text[:REMOVE_TOP_N_TXT]

    def __getitem__(self, index):
        if self.bucket_size > 1:
            index = min(len(self.text) - self.bucket_size, index)
            if self.encode_on_fly:
                for i in range(index, index + self.bucket_size):
                    if type(self.text[i]) is str:
                        self.text[i] = self.tokenizer.encode(self.text[i])
            # Return a bucket
            return self.text[index:index + self.bucket_size]
        else:
            if self.encode_on_fly and type(self.text[index]) is str:
                self.text[index] = self.tokenizer.encode(self.text[index])
            return self.text[index]

    def __len__(self):
        return len(self.text)
