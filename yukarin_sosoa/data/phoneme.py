from abc import abstractmethod
from os import PathLike
from pathlib import Path
from typing import Self

import numpy


class BasePhoneme(object):
    phoneme_list: tuple[str, ...]
    num_phoneme: int
    space_phoneme: str

    def __init__(
        self,
        phoneme: str,
        start: float,
        end: float,
    ):
        self.phoneme = phoneme
        self.start = numpy.round(start, decimals=4)
        self.end = numpy.round(end, decimals=4)

    def __repr__(self):
        return f"Phoneme(phoneme='{self.phoneme}', start={self.start}, end={self.end})"

    def __eq__(self, o: object):
        return isinstance(o, BasePhoneme) and (
            self.phoneme == o.phoneme and self.start == o.start and self.end == o.end
        )

    def verify(self):
        assert self.start < self.end, f"{self} start must be less than end"
        assert self.phoneme in self.phoneme_list, f"{self} is not defined."

    @property
    def phoneme_id(self):
        return self.phoneme_list.index(self.phoneme)

    @property
    def duration(self):
        return self.end - self.start

    @property
    def onehot(self):
        array = numpy.zeros(self.num_phoneme, dtype=bool)
        array[self.phoneme_id] = True
        return array

    @classmethod
    def parse(cls, s: str):
        """
        >>> BasePhoneme.parse('1.7425000 1.9125000 o:')
        Phoneme(phoneme='o:', start=1.7425, end=1.9125)
        """
        words = s.split()
        return cls(
            start=float(words[0]),
            end=float(words[1]),
            phoneme=words[2],
        )

    @classmethod
    @abstractmethod
    def convert(cls, phonemes: list[Self]) -> list[Self]:
        pass

    @classmethod
    def verify_list(cls: type[Self], phonemes: list[Self]):
        assert phonemes[0].start == 0, f"{phonemes[0]} start must be 0."
        for phoneme in phonemes:
            phoneme.verify()
        for pre, post in zip(phonemes[:-1], phonemes[1:]):
            assert pre.end == post.start, f"{pre} and {post} must be continuous."

    @classmethod
    def loads_julius_list(cls, text: str, verify=True):
        phonemes = [cls.parse(s) for s in text.split("\n") if len(s) > 0]
        phonemes = cls.convert(phonemes)

        if verify:
            try:
                cls.verify_list(phonemes)
            except Exception:
                raise
        return phonemes

    @classmethod
    def load_julius_list(cls, path: PathLike, verify=True):
        try:
            phonemes = cls.loads_julius_list(Path(path).read_text(), verify=verify)
        except Exception:
            print(f"{path} is not valid.")
            raise
        return phonemes

    @classmethod
    def save_julius_list(cls, phonemes: list[Self], path: PathLike, verify=True):
        if verify:
            try:
                cls.verify_list(phonemes)
            except Exception:
                print(f"{path} is not valid.")
                raise

        text = "\n".join(
            [
                f"{numpy.round(p.start, decimals=4):.4f}\t"
                f"{numpy.round(p.end, decimals=4):.4f}\t"
                f"{p.phoneme}"
                for p in phonemes
            ]
        )
        Path(path).write_text(text)


class OjtPhoneme(BasePhoneme):
    phoneme_list = (
        "pau",
        "A",
        "E",
        "I",
        "N",
        "O",
        "U",
        "a",
        "b",
        "by",
        "ch",
        "cl",
        "d",
        "dy",
        "e",
        "f",
        "g",
        "gw",
        "gy",
        "h",
        "hy",
        "i",
        "j",
        "k",
        "kw",
        "ky",
        "m",
        "my",
        "n",
        "ny",
        "o",
        "p",
        "py",
        "r",
        "ry",
        "s",
        "sh",
        "t",
        "ts",
        "ty",
        "u",
        "v",
        "w",
        "y",
        "z",
    )
    num_phoneme = len(phoneme_list)
    space_phoneme = "pau"

    @classmethod
    def convert(cls, phonemes: list[Self]):
        if "sil" in phonemes[0].phoneme:
            phonemes[0].phoneme = cls.space_phoneme
        if "sil" in phonemes[-1].phoneme:
            phonemes[-1].phoneme = cls.space_phoneme
        return phonemes
