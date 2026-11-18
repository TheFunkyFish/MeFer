import requests
import json


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
                assistant_response = data["choices"][0].get("text", "").strip()
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
