# Perplexity Calculation
Evaluate PodGPT and baseline using the Perplexity metric

## 📖 Download dataset
Download the audio and transcripts from the Objective Structured Clinical Examination (OSCE)-formatted medical conversational dataset:

Paper: [A dataset of simulated patient-physician medical interviews with a focus on respiratory cases](https://www.nature.com/articles/s41597-022-01423-1)

Dataset: [https://springernature.figshare.com/collections/A_dataset_of_simulated_patient-physician_medical_interviews_with_a_focus_on_respiratory_cases/5545842/1](https://springernature.figshare.com/collections/A_dataset_of_simulated_patient-physician_medical_interviews_with_a_focus_on_respiratory_cases/5545842/1)

## 🚀 Calculate perplexity
1. Please download PodGPT models via our script: [https://github.com/vkola-lab/PodGPT/blob/main/download_files/download_model_from_hf.py](https://github.com/vkola-lab/PodGPT/blob/main/download_files/download_model_from_hf.py).
2. Modify [this line](https://github.com/vkola-lab/PodGPT/blob/main/perplexity/calculate_perplexity.py#L233) to use your own Hugging Face Read token.
4. Run the following scripts to evaluate the specific model:

```shell
# Evaluate the baseline models
python calculate_perplexity.py --evaluate baseline --model google/gemma-2b-it
python calculate_perplexity.py --evaluate baseline --model google/gemma-7b-it
python calculate_perplexity.py --evaluate baseline --model shuyuej/Llama-3.3-70B-Instruct-GPTQ
python calculate_perplexity.py --evaluate baseline --model mistralai/Mixtral-8x7B-Instruct-v0.1
python calculate_perplexity.py --evaluate baseline --model meta-llama/Llama-3.3-70B-Instruct

# Evaluate PodGPT models
python download_model_from_hf.py shuyuej/gemma-2b-it-2048
python calculate_perplexity.py --evaluate PodGPT --model shuyuej/gemma-2b-it-2048 --id 9456 18912 28368 37824 47280

python download_model_from_hf.py shuyuej/gemma-7b-it-2048
python calculate_perplexity.py --evaluate PodGPT --model shuyuej/gemma-7b-it-2048 --id 18912 37824 56736 75648 94560

python download_model_from_hf.py shuyuej/PodGPT-v0.1
python calculate_perplexity.py --evaluate PodGPT --model shuyuej/Llama-3.3-70B-Instruct-GPTQ --lora True --id 18912 37824 56736 75648 94560

python download_model_from_hf.py shuyuej/Mixtral-8x7B-Instruct-v0.1-2048
python calculate_perplexity.py --evaluate PodGPT --model mistralai/Mixtral-8x7B-Instruct-v0.1 --lora True --id 20481 40962 61443 81924 102405

python download_model_from_hf.py shuyuej/Llama-3.3-70B-Instruct-2048
python calculate_perplexity.py --evaluate PodGPT--model meta-llama/Llama-3.3-70B-Instruct --lora True --id 18640
```

## ⛏️ Issues
1. As indicated in [this issue](https://github.com/huggingface/transformers/issues/29250), a `BOS` token is required to calculate perplexity for the Gemma 7B model. The solution can be found [here](https://github.com/huggingface/transformers/issues/29250#issuecomment-1966149282). Please be sure to uncomment [this line](https://github.com/vkola-lab/PodGPT/blob/main/perplexity/calculate_perplexity.py#L148) while evaluating the Gemma 7B model.
