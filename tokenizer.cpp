#include "tokenizer.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstdio>
#include <cctype>

bool Tokenizer::compareTokens(const TokenIndex& a, const TokenIndex& b) {
    return a.str < b.str;
}

Tokenizer::Tokenizer(const std::string& tokenizer_path, int vocab_size)
        : vocab_size(vocab_size) {
    vocab.resize(vocab_size);
    vocab_scores.resize(vocab_size);

    for (int i = 0; i < 256; i++) {
        byte_pieces[i * 2] = static_cast<char>(i);
        byte_pieces[i * 2 + 1] = '\0';
    }

    std::ifstream file(tokenizer_path, std::ios::binary);
    if (!file) {
        std::cerr << "Couldn't load " << tokenizer_path << std::endl;
        exit(EXIT_FAILURE);
    }

    file.read(reinterpret_cast<char*>(&max_token_length), sizeof(int));

    for (int i = 0; i < vocab_size; i++) {
        file.read(reinterpret_cast<char*>(&vocab_scores[i]), sizeof(float));
        int len;
        file.read(reinterpret_cast<char*>(&len), sizeof(int));
        vocab[i].resize(len);
        file.read(&vocab[i][0], len);
    }

    buildSortedVocab();
}

void Tokenizer::buildSortedVocab() {
    sorted_vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i] = {vocab[i], i};
    }
    std::sort(sorted_vocab.begin(), sorted_vocab.end(), compareTokens);
}

std::string Tokenizer::decode(int prev_token, int token) {
    std::string piece = vocab[token];
    if (prev_token == 1 && piece[0] == ' ') {
        piece = piece.substr(1);
    }

    unsigned char byte_val;
    if (sscanf(piece.c_str(), "<0x%2hhX>", &byte_val) == 1) {
        piece = std::string(byte_pieces.data() + byte_val * 2, 1);
    }

    return piece;
}

void Tokenizer::safePrintf(const std::string& piece) {
    if (piece.empty()) return;
    if (piece.length() == 1) {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return;
        }
    }
    std::cout << piece;
}

int Tokenizer::strLookup(const std::string& str) {
    TokenIndex tok{str, 0};
    auto it = std::lower_bound(sorted_vocab.begin(), sorted_vocab.end(), tok,
                               [](const TokenIndex& a, const TokenIndex& b) {
                                   return a.str < b.str;
                               });
    if (it != sorted_vocab.end() && it->str == str) {
        return it->id;
    }
    return -1;
}

std::vector<int> Tokenizer::encode(const std::string& text, bool bos, bool eos) {
    std::vector<int> tokens;
    tokens.reserve(text.length() + 2);  // +2 for potential BOS and EOS tokens

    if (bos) tokens.push_back(1);

    if (!text.empty()) {
        int dummy_prefix = strLookup(" ");
        tokens.push_back(dummy_prefix);
    }

    std::string str_buffer;
    str_buffer.reserve(max_token_length * 2 + 1 + 2);

    for (size_t i = 0; i < text.length(); ++i) {
        if ((text[i] & 0xC0) != 0x80) {
            str_buffer.clear();
        }

        str_buffer += text[i];

        if (i + 1 == text.length() || (text[i + 1] & 0xC0) != 0x80 || str_buffer.length() >= 4) {
            int id = strLookup(str_buffer);
            if (id != -1) {
                tokens.push_back(id);
            } else {
                for (char c : str_buffer) {
                    tokens.push_back(static_cast<unsigned char>(c) + 3);
                }
            }
            str_buffer.clear();
        }
    }

    while (true) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            std::string merge_str = vocab[tokens[i]] + vocab[tokens[i + 1]];
            int id = strLookup(merge_str);
            if (id != -1 && vocab_scores[id] > best_score) {
                best_score = vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) break;

        tokens[best_idx] = best_id;
        tokens.erase(tokens.begin() + best_idx + 1);
    }

    if (eos) tokens.push_back(2);

    return tokens;
}