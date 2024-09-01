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
    std::vector<int8_t> q;  // quantized values
    std::vector<float> s;   // scaling factors
} QuantizedTensor;

void quantize(QuantizedTensor *qx, float* x, int n) {
    int num_groups = n / GS;
    float Q_MAX = 127.0f;

    for (int group = 0; group < num_groups; group++) {

        // find the max absolute value in the current group
        float wmax = 0.0;
        for (int i = 0; i < GS; i++) {
            float val = fabs(x[group * GS + i]);
            if (val > wmax) {
                wmax = val;
            }
        }

        // calculate and write the scaling factor
        float scale = wmax / Q_MAX;
        qx->s[group] = scale;

        // calculate and write the quantized values
        for (int i = 0; i < GS; i++) {
            float quant_value = x[group * GS + i] / scale; // scale
            int8_t quantized = (int8_t) round(quant_value); // round and clamp
            qx->q[group * GS + i] = quantized;
        }
    }
}

void dequantize(QuantizedTensor *qx, float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = qx->q[i] * qx->s[i / GS];
    }
}

void naive_matmul(float *out, const float *x, const float *y, int x_row, int x_col, int y_col)
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

void naive_matmul_quantized(vector<float> &out, const QuantizedTensor *x, const QuantizedTensor *y, int x_row, int x_col, int y_col)
{
    int N_x = x_row * x_col;
    int N_y = x_col * y_col;

    for (int i = 0; i < x_row; i++)
    {
        for (int j = 0; j < y_col; j++)
        {
            for (int k = 0; k < x_col; k++)
            {
                out[(i * y_col) + j] += x->q[(i * x_col) + k] * y->q[(y_col * k) + j] * x->s[((i * x_col) + k)/GS] * y->s[((y_col * k) + j)/GS];
            }
        }
    }
}



void softmax(float *out, float *input, int row, int col)
{
    //todo numerical stability
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

void rmsnorm(float* out, float* activations, float* g, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++){
        sum += activations[i]*activations[i];
    }
    sum = sqrt((sum / n) + 1e-5f);
    for (int i = 0; i < n; i++){
        out[i] = activations[i] * g[i] / sum;
    }
}
typedef struct {
    int dim; // transformer dimension
    int hidden_dim; // for ffn layers
    int n_layers; // number of layers
    int n_heads; // number of query heads
    int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size; // vocabulary size, usually 256 (byte-level)
    int seq_len; // max sequence length
} Config;

class State {
public:
    float* x;
    float* x_norm;
    vector<float> q;
    vector<float> k;
    vector<float> v;
    QuantizedTensor* x_quantized;

    State() = default;

    State(int dim){
        x = new float[dim ];
        x_norm = new float[dim];
        q = vector<float>(dim);
        k = vector<float>(dim);
        v = vector<float>(dim);
        x_quantized = new QuantizedTensor();
    }

    ~State() {
        delete[] x;
        delete[] x_norm;
        delete x_quantized;
    }

};
class Transformer {
private:
    Config config;
    State s;
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
    QuantizedTensor* init_quantized_tensors(void** ptr, int n, int size_each) {
        auto* res = new QuantizedTensor[n];
        int8_t* p_int8 = static_cast<int8_t*>(*ptr);
        float* p_float = reinterpret_cast<float*>(p_int8 + n * size_each);

        for (int i = 0; i < n; i++) {
            res[i].q.assign(p_int8, p_int8 + size_each);
            res[i].s.assign(p_float, p_float + size_each / GS);
            p_int8 += size_each;
            p_float += size_each / GS;
        }

        *ptr = static_cast<void*>(p_float); // Update the pointer
        return res;
    }
    void load_weights(const std::string& filepath) {
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

        s = State(config.dim);
    }

    void attention_head(){

    }

    std::vector<float> forward(const float* token_embedding, int T){
        int hs = config.dim / config.n_heads;
        std::copy(token_embedding, token_embedding + config.dim, s.x);

        for (int layer=0; layer<config.n_layers; layer++){

            //rmsnorm
            rmsnorm(s.x_norm, s.x, attn_norm+config.dim*layer,config.dim);

            //extract q,k,v
            quantize(s.x_quantized, s.x,config.dim);
            naive_matmul_quantized(s.q, s.x_quantized, &wq[layer], 1, config.dim, config.dim);
            naive_matmul_quantized(s.k, s.x_quantized, wk + layer, 1, config.dim, config.n_kv_heads * hs);
            naive_matmul_quantized(s.v, s.x_quantized, wv + layer, 1, config.dim, config.n_kv_heads * hs);

            // rope

            for (int head=0; head<hs; head++){
                attention_head();
            }
            // cache state (1)
            // attn norm
            // attention
            // add attention output with (1) (residual)

            // cache state (2)
            // fnn norm
            // w1 (SwiGLU) w2 (SwiGLU) w3 (SwiGLU)
            // add output with (2) (residual)
        }
        // norm
        // wcls
        // softmax


    }

    int sample(std::vector<float> logits){
        return 2;
    }
    void init_state(){
        s.x = new float[config.dim];
        s.x_norm = new float[config.dim];
        s.q = vector<float>(config.dim);
        s.k = vector<float>(config.dim);
        s.v = vector<float>(config.dim);

        s.x_quantized = new QuantizedTensor();
        s.x_quantized->q = vector<int8_t>(config.dim);
        s.x_quantized->s = vector<float>(config.dim / GS);
    }

    void generate(const std::vector<int>& tokens, int max_step){
        init_state();
        //embedding
        std::vector<float> embedded(0);
        auto C = config.dim;

        int t = 0;
        int next = tokens[0];

        while (t<max_step) {
            std::cout<<next;

            const float* current_token_embedding = token_embeddings +  next * config.dim; //(4096, )
//            std::copy(start, start + config.dim, embedded.data() + t * config.dim);
//            embedded.insert(embedded.end(), start, start + config.dim);

            auto logits = forward(current_token_embedding, t+1);

            if (t < tokens.size()-1){
                next = tokens[t+1];
            } else {
                next = sample(logits);
            }
            if (next == 1){ // EOS predicted
                break;
            }
            t++;
        }
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
    transformer->generate(tokens, 256);

}
#endif
