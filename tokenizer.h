// File: tokenizer.h

#pragma once

#include <vector>
#include <string>
#include <array>

class Tokenizer {
private:
    struct TokenIndex {
        std::string str;
        int id;
    };

    std::vector<std::string> vocab;
    std::vector<float> vocab_scores;
    std::vector<TokenIndex> sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    std::array<char, 512> byte_pieces;

    static bool compareTokens(const TokenIndex& a, const TokenIndex& b);
    void buildSortedVocab();
    int strLookup(const std::string& str);

public:
    Tokenizer(const std::string& tokenizer_path, int vocab_size);
    ~Tokenizer() = default;

    std::string decode(int prev_token, int token);
    static void safePrintf(const std::string& piece);
    std::vector<int> encode(const std::string& text, bool bos = true, bool eos = true);
};