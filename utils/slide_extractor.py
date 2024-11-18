import base64
from typing import Literal, List
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chat_models.base import BaseChatModel
from langchain_openai import ChatOpenAI


class SlideElementPlaceholder(BaseModel):
    number_placeholder: int = Field(
        default=0,
        description="The number of all placeholder or seperate elements in the input power point slide's image.",
    )


class SlideElement(BaseModel):
    element_type: Literal["title", "sub-title", "body", "media"]
    count: int = Field(
        default=0,
        description="The number of **seperate** element belog to this type within the slide image. Default value to 0 if the element is not exist.",
    )


class SlideStructure(BaseModel):
    all_elements: List[SlideElement] = Field(
        description="list of all elements presented and editable from input slide image."
    )


def extract_structure(model: BaseChatModel, image_path: str):
    slide_model = model.with_structured_output(SlideStructure)
    placeholder_model = model.with_structured_output(SlideElementPlaceholder)

    # open image and convert to base64
    with open(image_path, "rb") as file:
        image_data = base64.b64encode(file.read()).decode("utf-8")

    placeholder_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """A helpful assistant that can extract and count the number of separate elements (far from each others, different in type bold, italic, etc) or placeholders in an input image of a PowerPoint slide.

### Element Types to Extract:
1. Title: Clear, concise title summarizing the main topic.
2. Sub-title: A brief, informative sub-title providing additional context.
3. Body: Accurately identify and count all sections of detailed content that elaborate on the topic in a structured manner. There may be multiple body elements within a single slide, so ensure to count each one separately.
4. Media (Image): Include relevant visual elements such as images, which can be placed anywhere within the slide to enhance understanding."""
            ),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ]
    )

    # define prompt
    slide_extraction_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                """A helpful assistant that can look in the provided image of a PowerPoint slide, 
We have total **{number_placeholder} elements** in this image.
Your job is to do classification on each separate elements (far from each others, different in type bold, italic, etc) in this image to specific type and count number of element of each type.

### Element Types to Extract:
1. Title: Clear, concise title summarizing the main topic.
2. Sub-title: A brief, informative sub-title providing additional context.
3. Body: Accurately identify and count all sections of detailed content that elaborate on the topic in a structured manner. There may be multiple body elements within a single slide, so ensure to count each one separately.
4. Media (Image): Include relevant visual elements such as images, which can be placed anywhere within the slide to enhance understanding."""
            ),
            (
                "user",
                [
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/jpeg;base64,{image_data}"},
                    }
                ],
            ),
        ]
    )

    placeholder_result: SlideElementPlaceholder = (
        placeholder_prompt | placeholder_model
    ).invoke({"image_data": image_data})
    number_placeholder: int = placeholder_result.number_placeholder
    print(f"Finish Detect placeholder: {number_placeholder}!!!")

    if number_placeholder == 0:
        return {}

    slide_extraction_chain = slide_extraction_prompt | slide_model
    output = slide_extraction_chain.invoke(
        {"image_data": image_data, "number_placeholder": number_placeholder}
    )
    return output
