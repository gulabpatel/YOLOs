import os
import streamlit as st
from ultralytics import YOLO
from IPython.display import display, Image
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain

model_path = 'best.pt'


st.set_page_config(
    page_title="Recipe Generator",  
    page_icon="🍳",     
    layout="wide",      
    initial_sidebar_state="expanded"    
)

st.title("Recipe Generator :female-cook:")
st.subheader("Upload an image to generate a recipe for the detected objects.")
st.markdown(
        f"""
        <style>
            body {{
                background-color: darkgreen;
            }}
        </style>
        """,
        unsafe_allow_html=True,)

def predict_objects_and_generate_recipe(image_path, model_path, define_conf):
    os.environ['HUGGINGFACEHUB_API_TOKEN'] = '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'
    
    # YOLO model ile nesneleri tanıma
    model = YOLO(model_path)
    results = model.predict(source=image_path, conf=define_conf, save=True)
    names = model.names
    
    unique_names = set()
    for r in results:
        for c in r.boxes.cls:
            unique_names.add(names[int(c)])
    
    return unique_names

def generate_recipe_using_language_model(unique_names):
    template = """Question: {question}: """
    prompt = PromptTemplate(template=template, input_variables=["question"])
    
    llm_chain = LLMChain(prompt=prompt, 
                         llm=HuggingFaceHub(repo_id="flax-community/t5-recipe-generation", 
                                            model_kwargs={"temperature": 0.3, "max_length": 512}))
    
    question = ', '.join(unique_names)
    recipe = llm_chain.run(question)
    
    return recipe

image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25)

if image_file is not None:
    
    temp_dir = 'temp'
    os.makedirs(temp_dir, exist_ok=True)  
    
    image_path = os.path.join(temp_dir, image_file.name)  
    with open(image_path, 'wb') as f:
        f.write(image_file.read())
    
    unique_names = predict_objects_and_generate_recipe(image_path, model_path, conf_threshold)
    recipe = generate_recipe_using_language_model(unique_names)
    
    title = recipe.split('ingredients:')[0]
    ingredients = recipe.split('directions:')[0].split('ingredients:')[1].split('answer the call 1 do cook.')[0].split(' ')
    directions = (recipe.split('directions:')[1].split('add salt and pepper to taste.')[0])

    full_recipe = '{}\n\n\nIngredients :\n{}\nDirections :\n{}'.format(
        title.capitalize().title(),
        '\n'.join(['    {}'.format(ingredient).title() for ingredient in ingredients]),
        '    {}'.format(directions).capitalize()
    )

 

    
    st.image(image_path)
    
    st.markdown("""
                <style>
                .big-font {
                    font-size:60px !important;
                }
                </style>
                """, unsafe_allow_html=True)
    st.write("Recipe:")
    st.write('<p class="big-font">'+full_recipe+'</p>', unsafe_allow_html=True, fontsize = 100)