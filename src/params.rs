use crate::config::LlamaConfigJson;
use crate::tensor::Tensor;
use safetensors::{SafeTensors, View};
pub struct LLamaParams<T> {
    // token_id to embedding lookup table
    pub embedding_table: Tensor<T>, // (vocab_size, dim)
    // decoder layer
    pub rms_att_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub wq: Vec<Tensor<T>>,        // (n_heads * head_size, hidden_size) x layers
    pub wk: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wv: Vec<Tensor<T>>,        // (n_kv_heads * head_size, hidden_size) x layers
    pub wo: Vec<Tensor<T>>,        // (hidden_size, n_heads * head_size) x layers
    // ffn layer
    pub rms_ffn_w: Vec<Tensor<T>>, // (hidden_size, ) x layers
    pub w_up: Vec<Tensor<T>>,      // (intermediate_size, hidden_size) x layers
    pub w_gate: Vec<Tensor<T>>,    // (intermediate_size, hidden_size) x layers
    pub w_down: Vec<Tensor<T>>,    // (hidden_size, intermediate_size) x layers
    // output
    pub rms_out_w: Tensor<T>, // (hidden_size, )
    pub lm_head: Tensor<T>,   // (vocab_size, dim)
}

impl LLamaParams<f32> {
    pub fn from_safetensors(safetensor: &SafeTensors, config: &LlamaConfigJson) -> Self {
        // todo!("实现从safetensors文件的模型参数加载");
        let get_tensor = |name: &str| -> Tensor<f32> {
            let tensor_view = safetensor.tensor(name).expect(&format!("Tensor {} not found", name));
            let data = tensor_view.data();
            let len = tensor_view.data_len();
            let shape = tensor_view.shape().to_vec();

            let mut data_vec = vec![0.0; len];
            for i in 0..len {
                data_vec[i] = data[i] as f32;
            }

            Tensor::new(data_vec, &shape)
        };

        let num_layers = config.num_hidden_layers as usize;
        let num_heads = config.num_attention_heads as usize;

        LLamaParams {
            embedding_table: get_tensor("model.embed_tokens.weight"),
            lm_head: if config.tie_word_embeddings {
                get_tensor("model.embed_tokens.weight")  // 复用 embedding_table
            } else {
                get_tensor("lm_head.weight")
            },
            rms_att_w: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.input_layernorm.weight", i))).collect(),
            rms_ffn_w: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i))).collect(),
            rms_out_w: get_tensor("model.norm.weight"),
            w_down: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i))).collect(),
            w_up: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i))).collect(),
            w_gate: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i))).collect(),
            wq: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i))).collect(),
            wk: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i))).collect(),
            wv: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i))).collect(),
            wo: (0..num_layers).map(|i| get_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i))).collect(),
        }
    }
}
