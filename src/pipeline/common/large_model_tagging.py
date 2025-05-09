import json
from typing import List


class LargeModelForTagging:
    def correct_response_format(self, text: str) -> List[str] | None:
        """
        Checks if there is a substring within the given text that starts with '[' and ends with ']'
        and follows the exact format:

        ["phrase", "phrase", "phrase", ...]

        If the format is correct, returns the JSON substring as a List[str]. Otherwise, returns None.
        """
        # Find indices of the last '[' and the last ']'
        start_index = text.rfind('[')
        end_index = text.rfind(']')

        # If we can't find a proper pair of brackets, return None
        if start_index == -1 or end_index == -1 or end_index < start_index:
            return None

        # Extract the substring that includes the brackets
        substring = text[start_index:end_index + 1]

        # If there's a "/" character in the substring, it's not a valid JSON
        if "/" in substring:
            return None

        # Delete all backslashes and underscores from the substring
        substring = substring.replace("\\", "")
        substring = substring.replace("_", " ")

        # Attempt to parse the substring as JSON
        try:
            parsed_data = json.loads(substring)
        except json.JSONDecodeError:
            return None

        # The parsed data must be a list
        if not isinstance(parsed_data, list):
            return None
        
        # Check each element in the list
        for element in parsed_data:
            # Each element must be a string
            if not isinstance(element, str):
                return None

        return parsed_data

    def response_fusion(self, responses: List[List[str]]) -> List[str]:
        """
        Merges the responses of multiple iterations into a single list,
        ensuring unique string values.
        """
        final_response = []
        unique_values = set()

        # Collect all unique values from the lists
        for response in responses:
            for value in response:
                if value not in unique_values:
                    unique_values.add(value)
                    final_response.append(value)

        print(f"Done. Merged response substring:\n\n", json.dumps(final_response, indent=4))

        return final_response

    def filter_banned_words(self, response: List[str], banned_words: List[str], ban_exact_word: bool = False) -> List[str]:
        """
        Remove banned words from the response.
        - If ban_exact_word is True, only exact matches are removed.
        - If ban_exact_word is False, any occurrence of the banned word in the string is removed.
        """
        print(f"{self.STR_PREFIX} Removing banned words...", flush=True)

        for banned_word in banned_words:
            for element in response:
                if (ban_exact_word and banned_word == element) or (not ban_exact_word and banned_word in element):
                    print(f"{self.STR_PREFIX} Discarded keyword: {element} (banned word: {banned_word})")
                    response = [value for value in response if banned_word not in value]
                    break

        return response
    
    def filter_long_values(self, response: List[str], max_words: int = 2) -> List[str]:
        """
        Remove values with more than a certain number of words.
        """
        print(f"{self.STR_PREFIX} Removing values with more than {max_words} words...", flush=True)

        result = []
    
        for element in response:
            # Check if the element has less than or equal to max_words
            if len(element.split()) <= max_words:
                # If it does, include it in the result
                result.append(element)
            else:
                # If it has more than max_words, print a message
                print(f"{self.STR_PREFIX} Discarded long value: {element}")

        return result
    
    def filter_redundant_substrings(self, response: List[str]) -> List[str]:
        """
        Remove redundant substrings from the response.
        
        A substring is considered redundant if it is contained within another string in the list.
        """
        print(f"{self.STR_PREFIX} Removing redundant substrings...", flush=True)
        
        # Set to store values that are substrings of other values
        redundant_elements = set()
        
        # Compare each value with every other value
        for element_i in response:
            for element_j in response:
                if element_i != element_j and element_i in element_j:
                    print(f"{self.STR_PREFIX} Discarded redundant substring: {element_j} (contains '{element_i}')")
                    redundant_elements.add(element_j)
        
        # Build the result dictionary, excluding redundant values
        return [value for value in response if value not in redundant_elements]

    def filter_duplicate_plurals(self, response: List[str]) -> List[str]:
        """
        Remove duplicate plural words from the response if their 
        singular form exists in the list.
        
        A word is considered plural if it ends with "s" or "es".
        """
        print(f"{self.STR_PREFIX} Removing duplicate plural words...", flush=True)

        # Step 1: Collect all individual words from all elements
        all_words = set()
        for element in response:
            all_words.update(element.split())

        unique_elements = set()

        # Step 2: Process each element in the list
        for element in response:
            words_in_element = element.split()
            filtered_words = []

            for word in words_in_element:
                original_word = word
                keep_word = True

                # Check if removing "es" results in a word that exists in all_words
                if original_word.endswith("es"):
                    singular_es = original_word[:-2]
                    if singular_es in all_words:
                        print(f"{self.STR_PREFIX} Discarded plural word: {original_word}")
                        keep_word = False

                # If "es" was not removed, check if removing "s" results in a word that exists in all_words
                if keep_word and original_word.endswith("s"):
                    singular_s = original_word[:-1]
                    if singular_s in all_words:
                        print(f"{self.STR_PREFIX} Discarded plural word: {original_word}")
                        keep_word = False

                if keep_word:
                    filtered_words.append(original_word)

            # Reconstruct the filtered element
            filtered_element = ' '.join(filtered_words)
            if filtered_element:
                unique_elements.add(filtered_element)

        return list(unique_elements)
