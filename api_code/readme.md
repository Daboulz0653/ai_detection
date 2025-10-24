# API_prompt_code

This folder contains the code necessary to generate synthetic fiction paragraphs from the OpenAI API. 

## Usage

This script (naturally) requires an API key to run.  Your API key can be found on the OpenAI platform [OpenAI platform](https://platform.openai.com/settings/profile), under My Profile -> User API keys. All generation that took place over Summer 2024 (10k+ paragraph) used less than five dollars. Charge your account accordingly. 

Add your OpenAI API key in the appropriate location. 

```
client = OpenAI(
  api_key = #your API key
)
```

### Program Inputs

The program takes in a sheet with a singular column filled with the prompts that you would like to prompt chatGPT with placed in the first column. 
```

### Program Outputs

Output: The program first parses the sheet and extract all prompts, placing them into a list. It then iterates over the list, repeatedly prompting chatGPT and placing the original prompt and the text response in a separate sheet called **{your sheet}_responses.csv**, with timestamps and mode (API or CHAT).

Note: the data we generated using the API code and analyzed for our paper (https://doi.org/10.1162/99608f92.6d5fb5ef) is contained in the whole_corpus.csv file. If this API code is used to regenerate the synthetic text corpus, note that the outputs will vary as GPT is a probabalistic model.



