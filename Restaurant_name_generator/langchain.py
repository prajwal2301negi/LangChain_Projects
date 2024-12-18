import os

from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.sequential import SequentialChain
from secret_key import openai_key

# Set the OpenAI API Key
os.environ['OPENAI_API_KEY'] = openai_key

# Initialize the LLM
llm = OpenAI(temperature=0.8)


def generate_restaurant_name_and_items(cuisine):

    # Prompt template for generating the restaurant name
    prompt_template_name = PromptTemplate(
        input_variables=['cuisine'],
        template="I want to open a restaurant for {cuisine} food. Suggest a fancy name for it."
    )
    
    # Chain for generating the restaurant name
    name_chain = LLMChain(llm=llm, prompt=prompt_template_name, output_key="restaurant_name")
    
    # Prompt template for generating menu items
    prompt_template_items = PromptTemplate(
        input_variables=['restaurant_name'],
        template="Suggest some menu items for {restaurant_name}. Return it as comma separated."
    )
    
    # Chain for generating the menu items
    food_items_chain = LLMChain(llm=llm, prompt=prompt_template_items, output_key="menu_items")
    
    # Sequential Chain to combine both chains
    overall_chain = SequentialChain(
        chains=[name_chain, food_items_chain],
        input_variables=['cuisine'],
        output_variables=['restaurant_name', 'menu_items']
    )
    
    # Run the overall chain
    response = overall_chain({'cuisine': cuisine})
    return response

# Testing
if __name__ == "__main__":
    cuisine = "Indian"
    result = generate_restaurant_name_and_items(cuisine)
    print("Restaurant Name:", result['restaurant_name'])
    print("Menu Items:", result['menu_items'])
