#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <cstdint>     // For fixed-width integer types
#include <cstdlib>     // For exit, EXIT_FAILURE
#include <cstdio>      // For fopen, fclose, etc.
#include <fcntl.h>     // For open flags like O_RDONLY
#include <unistd.h>    // For close
#include <sys/mman.h>  // For mmap
#include <sys/types.h> // For ssize_t
#include <sys/stat.h>  // For open
#include <cstring>     // For memset
#include <algorithm>
#include <fstream>
#include "tokenizer.h"

using namespace std;
int GS = 0;

typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

typedef struct {
    int8_t* q;    // quantized values
    float* s; // scaling factors
} QuantizedTensor;

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void naive_matmul(float *out, float *x, float *y, int x_row, int x_col, int y_row, int y_col)
{
    for (int i = 0; i < x_row; i++)
    {
        for (int j = 0; j < y_col; j++)
        {
            for (int k = 0; k < x_col; k++)
            {
                out[(i * y_col) + j] += x[(i * x_col) + k] * y[(y_col * k) + j];
            }
        }
    }
}


void softmax(float *out, float *input, int row, int col)
{

    for (int i = 0; i < row; i++)
    {
        // for every row, sum the exp of items
        float exp_sum = 0;
        auto* exp_mem = new float[col]();

        for (int j = 0; j < col; j++)
        {
            exp_mem[j] = expf(input[(i*col)+j]);
            exp_sum += expf(input[(i*col)+j]);
        }
        for (int j = 0; j < col; j++)
        {
            out[(i*col)+j] = exp_mem[j]/exp_sum;
        }
        delete[] exp_mem;
    }

}

void self_attn(float *out, float *K, float *Q, float *V, int T, int C, int head_size)
{
}

void attention_forward(float* out, float* input, int B, int T, int C, int head_size){

}
void attention_layer(float *out, float *input, int T, int C, int head_size){
    // generate K, Q, V from input
    float *K;
    float *Q;
    float *V; // (T,C)
    float *Wk;
    float *Wq;
    float *Wv;                              // (T,)
    naive_matmul(K, Wk, input, 3, 4, 4, 5); // (C, C) x (C, T), where inner dim should be divided into heads.
    naive_matmul(Q, Wq, input, 3, 4, 5, 6);
    naive_matmul(V, Wv, input, 3, 4, 5, 6);
    // self_attn(out, K, Q, V);
}


class Transformer {
private:
    float* attn_norm;
    float* fnn_norm;
    float* norm;
    float* token_embeddings;
    QuantizedTensor *q_tokens;
    QuantizedTensor *wq;
    QuantizedTensor *wk;
    QuantizedTensor *wv;
    QuantizedTensor *wo;
    QuantizedTensor *w1;
    QuantizedTensor *w2;
    QuantizedTensor *w3;
    QuantizedTensor *w_cls;

public:
    QuantizedTensor *init_quantized_tensors(void **ptr, int n, int size_each) {
        void *p = *ptr;
        auto *res = new QuantizedTensor[n * sizeof(QuantizedTensor)];
        for (int i = 0; i < n; i++) {
            /* map quantized int8 values */
            res[i].q = (int8_t*)p;
            p = (int8_t*)p + size_each;
            /* map scale factors */
            res[i].s = (float*)p;
            p = (float*)p + size_each / GS;
        }
        *ptr = p; // advance ptr to current position
        return res;
    }

    void load_weights(const std::string& filepath) {
        Config config;
        FILE *file = fopen(filepath.c_str(), "rb");
        if (!file) {
            std::cerr << "Couldn't open file " << filepath << std::endl;
            exit(EXIT_FAILURE);
        }

        // read in magic number (uint32), has to be 0x616b3432, i.e. "ak42" in ASCII
        uint32_t magic_number;
        if (fread(&magic_number, sizeof(uint32_t), 1, file) != 1) { exit(EXIT_FAILURE); }
        if (magic_number != 0x616b3432) { fprintf(stderr, "Bad magic number\n"); exit(EXIT_FAILURE); }

        // read in the version number (uint32), has to be 2
        int version;
        if (fread(&version, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
        if (version != 2) { fprintf(stderr, "Bad version %d, need version 2\n", version); exit(EXIT_FAILURE); }

        int header_size = 256; // the header size for version 2 in bytes

        // read in the Config
        if (fread(&config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

        // read in flags
        uint8_t shared_classifier; // a byte to indicate if the classifier is shared
        if (fread(&shared_classifier, sizeof(uint8_t), 1, file) != 1) { exit(EXIT_FAILURE); }

        int group_size; // the group size used in quantization
        if (fread(&group_size, sizeof(int), 1, file) != 1) { exit(EXIT_FAILURE); }
        GS = group_size; // set as global, as it will be used in many places

        // figure out the file size
        fseek(file, 0, SEEK_END); // move file pointer to end of file
        ssize_t file_size = ftell(file); // get the file size, in bytes
        fclose(file);

        // memory map the transformer weights into the data pointer
        int fd = open(filepath.c_str(), O_RDONLY); // open in read only mode
        if (fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
        float* data = static_cast<float*>(mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0));
        if (data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }

        void* ptr = ((char*) data) + header_size; // skip header bytes. char is 1 byte

        int head_size = config.dim / config.n_heads;

        // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
        auto* fptr = (float*) ptr; // cast our pointer to float*
        attn_norm = fptr;
        fptr += config.n_layers * config.dim;
        fnn_norm = fptr;
        fptr += config.n_layers * config.dim;
        norm = fptr;
        fptr += config.dim;

        // now read all the quantized weights
        ptr = (void*)fptr; // now cast the pointer back to void*
        q_tokens = init_quantized_tensors(&ptr, 1, config.vocab_size * config.dim);

        // dequantize token embedding table
        token_embeddings = new float[config.vocab_size * config.dim];
        dequantize(q_tokens, token_embeddings, config.vocab_size * config.dim);

        wq = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_heads * head_size));
        wk = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        wv = init_quantized_tensors(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size));
        wo = init_quantized_tensors(&ptr, config.n_layers, (config.n_heads * head_size) * config.dim);
        w1 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);
        w2 = init_quantized_tensors(&ptr, config.n_layers, config.hidden_dim * config.dim);
        w3 = init_quantized_tensors(&ptr, config.n_layers, config.dim * config.hidden_dim);

        w_cls = shared_classifier ? q_tokens : init_quantized_tensors(&ptr, 1, config.dim * config.vocab_size);
    }
    void forward(std::vector<int> tokens){

    }
};

#ifndef TESTING
int main()
{
    std::string prompt = "Hello!";

    Tokenizer tokenizer("../llama2_model_weights/tokenizer.bin", 32000);

    std::vector<int> tokens = tokenizer.encode(prompt);


    auto* transformer = new Transformer();
    transformer->load_weights("../llama2_model_weights/model.bin");
    transformer->forward(tokens);

}
#endif
