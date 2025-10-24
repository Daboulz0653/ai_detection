# [HDW_GPT_project](../DOCS.md)/[corpus_generation](../../DOCS.md)/API_prompt_code
This folder contains the code necessary to generate synthetic paragraphs from the API. 

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

This can be run either through an IDE by adding the sheet name into the script parameters in the program configurations, or can be run with the terminal using the following command: 

```
$ python3 prompt_runner example.csv
```

### Program Outputs

Output: The program first parses the sheet and extract all prompts, placing them into a list. It then iterates over the list, repeatedly prompting chatGPT and placing the original prompt and the text response in a separate sheet called **{your sheet}_responses.csv**, with timestamps and mode (API or CHAT).


## Things to Improve

1. When we were first writing the program, we decided to read in a csv and store the output in a separate csv file. The alternative is to do this using pandas and updating the data frame with the generated responses. This would avoid any accidental overwriting of generated data in cases of program failure. 

2. After experimentation with temperature and top_p values, we found that simply using the phrase "Assistant is a large language model trained by OpenAI”  yielded results closer to the online chat’s responses. This phrase was one used by Microsoft Azure and other online API tutorials. We decided to move forward with it (as opposed to adjusted temperature and top_p values) because there seems to be some precedence to using it in different applications.


