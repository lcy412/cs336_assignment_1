from cs336_basics.src.bpe_tokenizer.train import train_bpe_tokenizer
import cProfile

def main():

    train_bpe_tokenizer(
        input_file="tests/fixtures/corpus.en",
        vocab_size=500,
        special_tokens=["<|endoftext|>"],
    )

if __name__ == "__main__":
    cProfile.run("main()","bpe_profile.prof")