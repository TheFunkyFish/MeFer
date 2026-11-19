import requests
import json
import re
class LLMOutputCleaner:
    """
    A class to extract and clean text from LLM outputs wrapped in markdown code blocks.
    
    Handles:
    - Extraction of content between ``` markers.
    - Removal of language identifier after opening ``` (e.g., 'python', 'json').
    - Replacement of literal '\\n' sequences with actual newlines.
    """

    def __init__(self):
        # Regex to match the content inside triple backticks.
        # It captures everything between the first ``` and the next ```.
        # It allows for optional whitespace and language tags after the opening ```.
        self.code_block_pattern = re.compile(
            r'```(?:\w+)?\s*\n(.*?)\n?```',
            re.DOTALL
        )

    def extract_and_clean(self, raw_output: str) -> str:
        """
        Extracts the code block content, removes the language tag,
        and replaces literal '\\n' with actual newlines.
        
        Args:
            raw_output (str): The raw string returned by the LLM.
            
        Returns:
            str: The cleaned content with actual newlines.
        """
        if not raw_output:
            return ""

        # Step 1: Try to find the code block
        match = self.code_block_pattern.search(raw_output)
        
        if match:
            # Extract the content group (group 1)
            content = match.group(1)
        else:
            # Fallback: If no code block found, assume the whole string is the content
            # This is a safety net, but you might want to raise an error instead.
            content = raw_output.strip()

        # Step 2: Clean up the content
        # Replace literal backslash-n (\n) with actual newline characters
        # Note: In Python strings, '\\n' represents the two characters '\' and 'n'
        cleaned_content = content.replace('\\n', '\n')

        # Optional: Strip leading/trailing whitespace if desired
        # cleaned_content = cleaned_content.strip()

        return cleaned_content

    def extract_only(self, raw_output: str) -> str:
        """
        Extracts the raw content without replacing literal \n.
        Useful if you want to inspect the raw extracted text.
        """
        if not raw_output:
            return ""
        
        match = self.code_block_pattern.search(raw_output)
        if match:
            return match.group(1)
        else:
            return raw_output.strip()
cleaner = LLMOutputCleaner()


def main():
    # Get user input
    context = ""
    system_prompt = input("Enter system prompt: ")
    while(1):
        user_message = input("Enter your message: ")

        # Build the context prompt by combining system and user prompts
        # The context prompt is a template that includes both system and user parts

        # Prepare the payload
        payload = {
            "prompt": user_message,
            "system": system_prompt,
            "context": context
        }

        # Set headers
        headers = {
            "Content-Type": "application/json"
        }

        # Make the POST request
        try:
            response = requests.post(
                "http://localhost:8003/v1/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            # Parse the response
            data = response.json()
            #print(f"{data}")
            # Extract the assistant response
            if "choices" in data and len(data["choices"]) > 0:
                assistant_response = cleaner.extract_and_clean(data["choices"][0].get("text", "").strip())
            else:
                assistant_response = ""
            print(f"{assistant_response}")
            # The context prompt is always returned with the assistant response appended
            context = data['context']

            # Replace literal \s
            context = context.replace("\\", "")

            #print("--- Response ---")
            #print(context)
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
        except json.JSONDecodeError:
            print("Failed to decode JSON response.")
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
