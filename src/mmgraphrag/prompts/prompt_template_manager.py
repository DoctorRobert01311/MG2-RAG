import os
import asyncio
import importlib.util
from string import Template
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field, asdict
import base64
from PIL import Image
import io

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {
            "system": "system",
            "user": "user",
            "assistant": "assistant",
        },
        metadata={
            "help": "Mapping from default roles in prompte template files to specific LLM providers' defined roles."
        },
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False,
        default_factory=dict,
        metadata={
            "help": "A dict from prompt template names to templates. A prompt template can be a Template instance or a chat history which is a list of dict with content as Template instance."
        },
    )

    def __post_init__(self) -> None:
        """
        Initialize the templates directory and load templates.
        """
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)

        self.templates_dir = os.path.join(package_dir, "templates")

        self._load_templates()

    def _load_templates(self) -> None:
        """
        Load all templates from Python scripts in the templates directory.
        """
        if not os.path.exists(self.templates_dir):
            logger.error(f"Templates directory '{self.templates_dir}' does not exist.")
            raise FileNotFoundError(
                f"Templates directory '{self.templates_dir}' does not exist."
            )

        logger.info(f"Loading templates from directory: {self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]

                try:
                    module_name = f"src.mmgraphrag.prompts.templates.{script_name}"
                    module = importlib.import_module(module_name)

                    if not hasattr(module, "prompt_template"):
                        logger.error(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )
                        raise AttributeError(
                            f"Module '{module_name}' does not define a 'prompt_template'."
                        )

                    prompt_template = module.prompt_template
                    logger.debug(f"Loaded template from {module_name}")

                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item
                        for item in prompt_template
                    ):
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(
                                item["role"], item["role"]
                            )
                            item["content"] = (
                                item["content"]
                                if isinstance(item["content"], Template)
                                else Template(item["content"])
                            )
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(
                            f"Invalid prompt_template format in '{module_name}.py'. Must be a Template or List[Dict]."
                        )

                    logger.debug(
                        f"Successfully loaded template '{script_name}' from '{module_name}.py'."
                    )

                except Exception as e:
                    logger.error(
                        f"Failed to load template from '{module_name}.py': {e}"
                    )
                    raise
    
    def _encode_image(self, image_input: Union[str, Any]) -> Optional[str]:
        """
        Helper to convert image input (path or PIL object) to base64 string.
        """
        try:
            if isinstance(image_input, str):
                if not os.path.exists(image_input):
                    logger.warning(f"Image path does not exist: {image_input}")
                    return None
                with open(image_input, "rb") as image_file:
                    return base64.b64encode(image_file.read()).decode('utf-8')

            elif Image and isinstance(image_input, Image.Image):
                buffered = io.BytesIO()
                if image_input.mode in ("RGBA", "P"):
                    image_input = image_input.convert("RGB")
                image_input.save(buffered, format="JPEG")
                return base64.b64encode(buffered.getvalue()).decode('utf-8')

            else:
                logger.warning(f"Unsupported image input type: {type(image_input)}")
                return None
        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def _inject_image_to_messages(
        self, messages: List[Dict[str, Any]], base64_image: str
    ) -> List[Dict[str, Any]]:
        """
        Injects the base64 image into the 'user' message, converting content to List format.
        """
        new_messages = []
        image_injected = False
        
        for idx, msg in enumerate(messages):
            if msg["role"] == self.role_mapping.get("user", "user") and not image_injected and idx == len(messages) - 1:
                new_content = [
                    {"type": "text", "text": msg["content"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                    }
                ]
                new_msg = {"role": msg["role"], "content": new_content}
                image_injected = True
                new_messages.append(new_msg)
            else:
                new_messages.append(msg)
        
        return new_messages
    
    def render(
        self, 
        name: str, 
        image_input: Optional[Union[str, Image.Image]] = None, 
        **kwargs
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Render a template. If `image_input` is provided, constructs a multimodal message.

        Args:
            name: Template name.
            image_input: Optional path to image or PIL.Image object.
            kwargs: Variables for text substitution. 包含 prompt_user
        """
        template = self.get_template(name)
        
        rendered_result = None
        if isinstance(template, Template):
            try:
                rendered_result = template.substitute(**kwargs)
            except KeyError as e:
                raise ValueError(f"Missing variable for template '{name}': {e}")
                
        elif isinstance(template, list):
            try:
                rendered_result = [
                    {
                        "role": item["role"],
                        "content": item["content"].substitute(**kwargs),
                    }
                    for item in template
                ]
            except KeyError as e:
                raise ValueError(f"Missing variable in template '{name}': {e}")

        if image_input is None:
            return rendered_result
        
        base64_img = self._encode_image(image_input)
        if not base64_img:
            logger.warning("Image encoding failed or image not found. Returning text-only prompt.")
            return rendered_result

        if isinstance(rendered_result, list):
            return self._inject_image_to_messages(rendered_result, base64_img)
        elif isinstance(rendered_result, str):
            logger.info(f"Converting string template '{name}' to chat format to attach image.")
            return [
                {
                    "role": self.role_mapping.get("user", "user"),
                    "content": [
                        {"type": "text", "text": rendered_result},
                        {
                            "type": "image_url", 
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}
                        }
                    ]
                }
            ]
            
        return rendered_result
    
    def sync_render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        return asyncio.run(self.render(name, **kwargs))

    def list_template_names(self) -> List[str]:
        """
        List all available template names.

        Returns:
            List[str]: A list of template names.
        """
        logger.info("Listing all available template names.")

        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        Retrieve a template by name.

        Args:
            name (str): The name of the template.

        Returns:
            Union[Template, List[Dict[str, Any]]]: The requested template.

        Raises:
            KeyError: If the template is not found.
        """
        if name not in self.templates:
            logger.error(f"Template '{name}' not found.")
            raise KeyError(f"Template '{name}' not found.")
        logger.debug(f"Retrieved template '{name}'.")

        return self.templates[name]

    def print_template(self, name: str) -> None:
        """
        Print the prompt template string or chat history structure for the given template name.

        Args:
            name (str): The name of the template.

        Raises:
            KeyError: If the template is not found.
        """
        try:
            template = self.get_template(name)
            print(f"Template name: {name}")
            if isinstance(template, Template):
                print(template.template)
            elif isinstance(template, list):
                for item in template:
                    print(f"Role: {item['role']}, Content: {item['content']}")
            logger.info(f"Printed template '{name}'.")
        except KeyError as e:
            logger.error(f"Failed to print template '{name}': {e}")
            raise

    def is_template_name_valid(self, name: str) -> bool:
        return name in self.templates
