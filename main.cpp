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
#include <memory>
#include <array>  // Add this at the top with other includes
#include "tokenizer.h"

class QuantizedTensor {
public:
    std::vector<int8_t> q;  // quantized values
    std::vector<float> s;   // scaling factors
    int group_size;         // group size for quantization

    QuantizedTensor() = default;

    QuantizedTensor(int size, int gs = 64) : group_size(gs) {
        q = std::vector<int8_t>(size);
        s = std::vector<float>(size / group_size);
    }

    static std::vector<QuantizedTensor> init_from_ptr(void** ptr, int n, int size_each, int gs) {
        auto res = std::vector<QuantizedTensor>(n);
        char* p = static_cast<char*>(*ptr);

        for (int i = 0; i < n; i++) {
            res[i].group_size = gs;
            // Map quantized int8 values
            int8_t* q_ptr = reinterpret_cast<int8_t*>(p);
            res[i].q.assign(q_ptr, q_ptr + size_each);
            p += size_each;

            // Map scale factors
            float* s_ptr = reinterpret_cast<float*>(p);
            res[i].s.assign(s_ptr, s_ptr + size_each / gs);
            p += size_each / gs * sizeof(float);
        }

        *ptr = p;  // Update the pointer
        return res;
    }

    void quantize(float* x, int n) {
        int num_groups = n / group_size;
        float Q_MAX = 127.0f;

        for (int group = 0; group < num_groups; group++) {
            // find the max absolute value in the current group
            float wmax = 0.0;
            for (int i = 0; i < group_size; i++) {
                float val = fabs(x[group * group_size + i]);
                if (val > wmax) {
                    wmax = val;
                }
            }

            // calculate and write the scaling factor
            float scale = wmax / Q_MAX;
            s[group] = scale;

            // calculate and write the quantized values
            for (int i = 0; i < group_size; i++) {
                float quant_value = x[group * group_size + i] / scale;
                int8_t quantized = (int8_t) round(quant_value);
                q[group * group_size + i] = quantized;
            }
        }
    }

    void dequantize(float* x, int n) const {
        for (int i = 0; i < n; i++) {
            x[i] = q[i] * s[i / group_size];
        }
    }
};

void naive_matmul(float *out, const float *x, const float *y, int x_row, int x_col, int y_col)
{
    // just for reference. multiplies 2 matrices.
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

void naive_matmul_quantized2(std::vector<float> & out, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // initially wrote this function, the output of logits is very slightly off (after 2 decimal places). 
    int GS_w = w->group_size;
    int GS_x = x->group_size; 
    for (int i = 0; i < d; i++) {
        float accumalate = 0.0f;
        int j;
        for (j = 0; j < n; j++) {
            accumalate += ((float) ((int32_t) x->q[j]) * ((int32_t) w->q[i * n + j])) * w->s[(i * n + j) / GS_w] * x->s[j / GS_x];
        }
        out[i] = accumalate;
    }
}

void naive_matmul_quantized(std::vector<float> & xout, QuantizedTensor *x, QuantizedTensor *w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    int i;
    int GS_w = w->group_size;
    int GS_x = x->group_size; 
    // #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {

        float val = 0.0f;
        int32_t ival = 0;
        int in = i * n;

        // do the matmul in groups of GS
        int j;
        for (j = 0; j <= n - GS_w; j += GS_w) {

            for (int k = 0; k < GS_w; k++) {
                ival += ((int32_t) x->q[j + k]) * ((int32_t) w->q[in + j + k]);

            }            
            val += ((float) ival) * w->s[(in + j) / GS_w] * x->s[j / GS_x];  
            ival = 0;
        }

        xout[i] = val;
    }
}

void softmax(std::vector<float> &out, const std::vector<float>& input, int row, int col) {
    for (int i = 0; i < row; i++) {
        // Find max value in current row for numerical stability
        float max_val = input[i * col];
        for (int j = 1; j < col; j++) {
            max_val = std::max(max_val, input[i * col + j]);
        }       

        // Compute exp(x - max) and sum
        float exp_sum = 0.0f;
        std::vector<float> exp_vals(col);
        for (int j = 0; j < col; j++) {
            exp_vals[j] = expf(input[i * col + j] - max_val);
            exp_sum += exp_vals[j];
        }

        // Normalize
        for (int j = 0; j < col; j++) {
            out[i * col + j] = exp_vals[j] / exp_sum;
        }
    }
}

void softmax_unstable(std::vector<float> &out, std::vector<float>input, int row, int col)
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

void rmsnorm(std::vector<float> &out, const std::vector<float> &activations, float* g, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++){
        sum += activations[i]*activations[i];
    }
    sum = 1.0f /sqrt((sum / n) + 1e-5f);
    for (int i = 0; i < n; i++){
        out[i] =  g[i] * (sum*activations[i]); //order matters for fp-op
    }
}
typedef struct {
    int dim;          // transformer dimension
    int hidden_dim;   // for ffn layers
    int n_layers;     // number of layers
    int n_heads;      // number of query heads
    int n_kv_heads;   // number of key/value heads (can be < query heads because of multiquery)
    int vocab_size;   // vocabulary size, usually 256 (byte-level)
    int seq_len;      // max sequence length
} Config;


class State {
public:
    std::vector<float> x;
    std::vector<float> x2;
    std::vector<float> x_swiglu;
    std::vector<float> x_swiglu2;
    std::vector<float> x_norm;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> logits;
    std::unique_ptr<QuantizedTensor> x_quantized;
    std::unique_ptr<QuantizedTensor> h_quantized;

    std::vector<std::vector<std::vector<float>>> k_cache;// (t, l, dim)
    std::vector<std::vector<std::vector<float>>> v_cache;

    State() = default;

    State(int dim, int hidden_dim, int vocab_size){
        x = std::vector<float>(dim);
        x_norm = std::vector<float>(dim);
        x2 = std::vector<float>(dim);
        x_swiglu = std::vector<float>(hidden_dim);
        x_swiglu2 = std::vector<float>(hidden_dim);

        q = std::vector<float>(dim);
        k = std::vector<float>(dim);
        v = std::vector<float>(dim);

        logits = std::vector<float>(vocab_size);

        x_quantized = std::make_unique<QuantizedTensor>(dim);
        h_quantized = std::make_unique<QuantizedTensor>(hidden_dim);
    }
};
class Transformer {
private:
    Config config;
    std::unique_ptr<State> s;
    std::vector<float> attn_norm;    
    std::vector<float> fnn_norm;     
    std::vector<float> norm;         
    std::vector<float> token_embeddings;  
    std::vector<QuantizedTensor> q_tokens;
    std::vector<QuantizedTensor> wq; //std::vector of size n_layer, each entry is a (dim, dim)
    std::vector<QuantizedTensor> wk;
    std::vector<QuantizedTensor> wv;
    std::vector<QuantizedTensor> wo;
    std::vector<QuantizedTensor> w1;
    std::vector<QuantizedTensor> w2;
    std::vector<QuantizedTensor> w3;
    std::vector<QuantizedTensor> w_cls;

public:
    void load_weights(const std::string& filepath) {
        //Karpathy's load weights.
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
        int GS = group_size; // set as global, as it will be used in many places

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
        attn_norm.assign(fptr, fptr + config.n_layers * config.dim);
        fptr += config.n_layers * config.dim;
        
        fnn_norm.assign(fptr, fptr + config.n_layers * config.dim);
        fptr += config.n_layers * config.dim;
        
        norm.assign(fptr, fptr + config.dim);
        fptr += config.dim;

        // now read all the quantized weights
        ptr = (void*)fptr; // now cast the pointer back to void*
        q_tokens = QuantizedTensor::init_from_ptr(&ptr, 1, config.vocab_size * config.dim, GS);

        // dequantize token embedding table
        token_embeddings.resize(config.vocab_size * config.dim);
        q_tokens[0].dequantize(token_embeddings.data(), config.vocab_size * config.dim);

        wq = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.dim * (config.n_heads * head_size), GS);
        wk = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size), GS);
        wv = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.dim * (config.n_kv_heads * head_size), GS);
        wo = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, (config.n_heads * head_size) * config.dim, GS);
        w1 = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.dim * config.hidden_dim, GS);
        w2 = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.hidden_dim * config.dim, GS);
        w3 = QuantizedTensor::init_from_ptr(&ptr, config.n_layers, config.dim * config.hidden_dim, GS);

        w_cls = shared_classifier ? q_tokens : QuantizedTensor::init_from_ptr(&ptr, 1, config.dim * config.vocab_size, GS);

    }

    void attention_head(std::vector<float>& att_out, const std::vector<float>& q, const std::vector<float>& k, const std::vector<float>& v, int layer, int T, int head) {
        //(128,) each 
        // loop on each token, multiply its k vec by current token's q vec. 
        // take the output vec of activations which is of size T, do softmax,
        // and elementwise softmax out with every token's v vec.
        int hs = q.size();
        std::vector<float> pre_soft (T, 0.0f);
        for (int t = 0; t < T; t++){
            for (int j = 0; j< hs; j++){
                pre_soft[t] += q[j] * s->k_cache[t][layer][hs * head + j];
            }
            pre_soft[t] /= sqrtf(hs);
        }

        std::vector<float> post_softmax (T);
        softmax(post_softmax, pre_soft, 1, T);

        for (int t = 0; t < T; t++){
            for (int j = 0; j< hs; j++){
                att_out[hs*head + j] += post_softmax[t] * s->v_cache[t][layer][hs * head + j];
            }
        }

    }

    std::vector<float> forward(int next, int t){ //rename to current.
        int hs = config.dim / config.n_heads;

        auto token_embedding_start = token_embeddings.begin() + next * config.dim;
        std::copy(token_embedding_start, token_embedding_start + config.dim, s->x.begin());

        // add spot for current token in cache
        s->k_cache.emplace_back(std::vector<std::vector<float>>(config.n_layers, std::vector<float>(config.dim, 0.0f)));
        s->v_cache.emplace_back(std::vector<std::vector<float>>(config.n_layers, std::vector<float>(config.dim, 0.0f)));


        for (int layer=0; layer<config.n_layers; layer++){

            //rmsnorm
            rmsnorm(s->x2, s->x, attn_norm.data() + config.dim * layer, config.dim);

            //extract q,k,v
            s->x_quantized->quantize(s->x2.data(), config.dim);
            naive_matmul_quantized(s->q, s->x_quantized.get(), &wq[layer], config.dim, config.dim); // (dim, )
            naive_matmul_quantized(s->k, s->x_quantized.get(), &wk[layer], config.dim, config.n_kv_heads * hs);
            naive_matmul_quantized(s->v, s->x_quantized.get(), &wv[layer], config.dim, config.n_kv_heads * hs);
            
            // rope
            for (int h = 0; h < config.n_heads; h++) {
                for (int i = 0; i < hs; i += 2) {
                    float theta = powf(10000.0f, (-i / (float)hs));
                    float cos_calc = cos(t * theta);
                    float sin_calc = sin(t * theta);

                    int qi = h * hs + i;
                    float q1 = s->q[qi];
                    float q2 = s->q[qi + 1];
                    s->q[qi] = q1 * cos_calc - q2 * sin_calc;
                    s->q[qi + 1] = q1 * sin_calc + q2 * cos_calc;
                    
                    if (h < config.n_kv_heads) {
                        float k1 = s->k[qi];
                        float k2 = s->k[qi + 1];
                        s->k_cache[t][layer][qi] = k1 * cos_calc - k2 * sin_calc;
                        s->k_cache[t][layer][qi + 1] = k1 * sin_calc + k2 * cos_calc;
                    }
                }

            }


            // s->k_cache[t][layer] = s->k; //no need to copy since we process it directly in rope.
            s->v_cache[t][layer] = s->v;

            std::vector<float> att_out (config.dim, 0.0f);

            for (int head=0; head<config.n_heads; head++){
                std::vector<float> q_head(s->q.begin() + head * hs, s->q.begin() + head * hs + hs); //(hs, )
                std::vector<float> k_head(s->k.begin() + head * hs, s->k.begin() + head * hs + hs);
                std::vector<float> v_head(s->v.begin() + head * hs, s->v.begin() + head * hs + hs);


                attention_head(att_out, q_head, k_head, v_head, layer, t+1, head);
            }



            s->x_quantized->quantize(att_out.data(), config.dim);
            naive_matmul_quantized(s->x2, s->x_quantized.get(), &wo[layer], config.dim, config.dim); 

            // residual layer
            for (int i = 0; i < config.dim; i++) {
                s->x[i] += s->x2[i];
            }

            // fnn norm
            rmsnorm(s->x2, s->x, fnn_norm.data() + config.dim * layer, config.dim);
            s->x_quantized->quantize(s->x2.data(), config.dim);
            
            naive_matmul_quantized(s->x_swiglu2, s->x_quantized.get(), &w1[layer], config.dim, config.hidden_dim);
            naive_matmul_quantized(s->x_swiglu, s->x_quantized.get(), &w3[layer], config.dim, config.hidden_dim);
            
            for (int i = 0; i < config.hidden_dim; i++) {
                float gate = s->x_swiglu2[i] * (1.0f / (1.0f + expf(-s->x_swiglu2[i]))); //silu(x)=x*sigm(x)
                s->x_swiglu2[i] = gate * s->x_swiglu[i];
            }

            s->h_quantized->quantize(s->x_swiglu2.data(), config.hidden_dim);
            naive_matmul_quantized(s->x2, s->h_quantized.get(), &w2[layer], config.hidden_dim, config.dim);


            for (int i = 0; i < config.dim; i++) {
                s->x[i] += s->x2[i];
            }

        }
        // norm
        rmsnorm(s->x, s->x, norm.data(), config.dim);

        // wcls
        s->x_quantized->quantize(s->x.data(), config.dim);
        naive_matmul_quantized(s->logits, s->x_quantized.get(), &w_cls[0], config.dim, config.vocab_size);

        return s->logits;
    }

    int sample(std::vector<float> logits){
        int max_indx = 0;
        float max_logit = logits[0];
        for (int i = 0; i < logits.size(); i++){
            float logit = logits[i];
            if (logit > max_logit){
                max_logit = logit;
                max_indx = i;
            }
        } 
        return max_indx;
    }


    void generate(const std::vector<int>& tokens, int max_step, Tokenizer tokenizer){
        s = std::make_unique<State>(config.dim, config.hidden_dim, config.vocab_size);

        int t = 0;
        int curr = tokens[0];
        int next;
        while (t<max_step) {
            auto logits = forward(curr, t);

            if (t < tokens.size()-1){ //-1 because we processed bos already first iter.
                next = tokens[t+1];
            } else {
                next = sample(logits);
            }
            if (next == 1){ // EOS predicted
                break;
            }
            std::string out = tokenizer.decode(curr, next);
            std::cout << out << std::flush;
            t++; 
            curr = next;
        }
    }
};

#ifndef TESTING
int main(int argc, char* argv[])
{
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <prompt> <tokenizer_path> <model_path>" << std::endl;
        return 1;
    }

    std::string prompt = argv[1];
    std::string tokenizer_path = argv[2];
    std::string model_path = argv[3];

    Tokenizer tokenizer(tokenizer_path, 32000);
    std::vector<int> tokens = tokenizer.encode(prompt, 1, 0);

    auto* transformer = new Transformer();
    transformer->load_weights(model_path);
    transformer->generate(tokens, 256, tokenizer);

    return 0;
}

#endif
