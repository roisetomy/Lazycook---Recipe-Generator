import os
import google.generativeai as genai
from typing import List, Dict, Any, Optional, Tuple
import json

class ShoppingListAgent:
    """
    A React-style shopping list agent that can perform multiple intelligent actions
    to manage a shopping list based on ingredients and user requests.
    """
    
    def __init__(self, shopping_list_file: str = None, api_key: str = None):
        """
        Initialize the shopping list agent.
        
        Args:
            shopping_list_file: Path to the shopping list file. If None, uses default location.
            api_key: Google API key. If None, uses environment variable.
        """
        # Configure SDK
        if api_key:
            genai.configure(api_key=api_key)
        else:
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        
        # Set shopping list file path
        if shopping_list_file:
            self.shopping_list_file = shopping_list_file
        else:
            # Default to one folder outside of current directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            self.shopping_list_file = os.path.join(parent_dir, "shopping_list.txt")
        
        # Load shopping list
        self.shopping_list = self._load_shopping_list()
        
        # Initialize function declarations
        self._setup_function_declarations()
        
        # System prompt for React-style reasoning
        self.system_prompt = """
        You are a smart shopping list agent that can perform multiple actions to intelligently manage a shopping list. 

        When a user gives you ingredients or items, you should:
        1. FIRST check what's already on the shopping list using check_items_exist()
        2. THEN decide what actions to take based on what you find:
           - If items already exist but user wants different quantities, use update_item_quantity()
           - If items don't exist, use add_items()
           - If user wants to remove items, use remove_items()
        3. ALWAYS explain what actions you took and why

        You can call multiple functions in sequence to accomplish complex tasks. Think step by step and be proactive about checking existing items before making changes.

        Key behaviors:
        - Always check existing items first before adding
        - Be smart about partial matches (e.g., "bread" matches "2 loaves of bread")
        - Update quantities when appropriate rather than adding duplicates
        - Provide clear explanations of what you did
        - Handle multiple items in one request efficiently
        """
    
    def _load_shopping_list(self) -> List[str]:
        """Load shopping list from file."""
        if os.path.exists(self.shopping_list_file):
            with open(self.shopping_list_file, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return []
    
    def _save_shopping_list(self):
        """Save shopping list to file."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.shopping_list_file), exist_ok=True)
        
        with open(self.shopping_list_file, "w", encoding="utf-8") as f:
            for item in self.shopping_list:
                f.write(item + "\n")
    
    def _get_shopping_list(self) -> Dict[str, Any]:
        """Return current items."""
        return {"shopping_list": self.shopping_list}
    
    def _add_items(self, items: List[str]) -> Dict[str, Any]:
        """Add items, ignoring duplicates."""
        added = []
        for item in items:
            if item not in self.shopping_list:
                self.shopping_list.append(item)
                added.append(item)
        self._save_shopping_list()
        return {"shopping_list": self.shopping_list, "added": added}
    
    def _remove_items(self, items: List[str]) -> Dict[str, Any]:
        """Remove items that exist; ignore unknowns."""
        removed = []
        for item in items:
            if item in self.shopping_list:
                self.shopping_list.remove(item)
                removed.append(item)
        self._save_shopping_list()
        return {"shopping_list": self.shopping_list, "removed": removed}
    
    def _update_item_quantity(self, item: str, new_quantity: str) -> Dict[str, Any]:
        """Update quantity of an existing item or add new item with quantity."""
        # Find existing item (case-insensitive, partial match)
        existing_item = None
        for existing in self.shopping_list:
            if item.lower() in existing.lower() or existing.lower() in item.lower():
                existing_item = existing
                break
        
        if existing_item:
            self.shopping_list.remove(existing_item)
            self.shopping_list.append(f"{new_quantity} {item}")
            self._save_shopping_list()
            return {"shopping_list": self.shopping_list, "updated": f"{existing_item} -> {new_quantity} {item}"}
        else:
            self.shopping_list.append(f"{new_quantity} {item}")
            self._save_shopping_list()
            return {"shopping_list": self.shopping_list, "added": f"{new_quantity} {item}"}
    
    def _check_items_exist(self, items: List[str]) -> Dict[str, Any]:
        """Check which items already exist on the shopping list."""
        existing = []
        missing = []
        
        for item in items:
            found = False
            for existing_item in self.shopping_list:
                if item.lower() in existing_item.lower() or existing_item.lower() in item.lower():
                    existing.append({"requested": item, "existing": existing_item})
                    found = True
                    break
            if not found:
                missing.append(item)
        
        return {"existing": existing, "missing": missing, "shopping_list": self.shopping_list}
    
    def _setup_function_declarations(self):
        """Setup function declarations for Gemini."""
        self.py_funcs = {
            "get_shopping_list": self._get_shopping_list,
            "add_items": self._add_items,
            "remove_items": self._remove_items,
            "update_item_quantity": self._update_item_quantity,
            "check_items_exist": self._check_items_exist,
        }
        
        self.function_declarations = [
            genai.protos.FunctionDeclaration(
                name="get_shopping_list",
                description="Returns the current items on the shopping list.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={}
                )
            ),
            genai.protos.FunctionDeclaration(
                name="check_items_exist",
                description="Check which items from a list already exist on the shopping list. Use this first before adding items.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "items": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(type=genai.protos.Type.STRING),
                            description="Items to check for existence"
                        )
                    },
                    required=["items"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="add_items",
                description="Add one or more items to the shopping list.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "items": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(type=genai.protos.Type.STRING),
                            description="Items to add"
                        )
                    },
                    required=["items"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="remove_items",
                description="Remove one or more items from the shopping list.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "items": genai.protos.Schema(
                            type=genai.protos.Type.ARRAY,
                            items=genai.protos.Schema(type=genai.protos.Type.STRING),
                            description="Items to remove"
                        )
                    },
                    required=["items"]
                )
            ),
            genai.protos.FunctionDeclaration(
                name="update_item_quantity",
                description="Update the quantity of an existing item or add a new item with specific quantity.",
                parameters=genai.protos.Schema(
                    type=genai.protos.Type.OBJECT,
                    properties={
                        "item": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The item name"
                        ),
                        "new_quantity": genai.protos.Schema(
                            type=genai.protos.Type.STRING,
                            description="The new quantity (e.g., '2 loaves of', '1 gallon of', '3 lbs')"
                        )
                    },
                    required=["item", "new_quantity"]
                )
            ),
        ]
    
    def process_ingredients(self, ingredients_to_buy: List[str], user_message: str = None) -> Tuple[str, List[str]]:
        """
        Process a list of ingredients to buy, checking against existing shopping list
        and making intelligent decisions about what to add/update.
        
        Args:
            ingredients_to_buy: List of ingredients that need to be purchased
            user_message: Optional custom message to provide context
            
        Returns:
            Tuple of (agent_response, updated_chat_history)
        """
        if not user_message:
            user_message = f"I need to buy these ingredients: {', '.join(ingredients_to_buy)}. Please check what's already on my shopping list and add what's missing."
        
        response, _ = self._react_agent(user_message)
        return response, []
    
    def chat(self, user_message: str, chat_history: List = None) -> Tuple[str, List]:
        """
        Chat with the shopping list agent.
        
        Args:
            user_message: User's message/request
            chat_history: Previous chat history (optional)
            
        Returns:
            Tuple of (agent_response, updated_chat_history)
        """
        return self._react_agent(user_message, chat_history)
    
    def _react_agent(self, user_text: str, history: List = None) -> Tuple[str, List]:
        """
        React-style agent that can perform multiple actions intelligently.
        """
        if history is None:
            history = []

        # Create the model with system prompt
        model = genai.GenerativeModel(
            'gemini-1.5-flash',
            tools=[genai.protos.Tool(function_declarations=self.function_declarations)],
            system_instruction=self.system_prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.2,
            )
        )

        # Start chat with history
        chat = model.start_chat(history=history)

        # Send user message
        response = chat.send_message(user_text)
        
        # Continue conversation until no more function calls
        while True:
            # Check if model wants to call functions
            function_calls = []
            if response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        function_calls.append(part.function_call)
            
            if function_calls:
                # Execute all function calls and prepare responses
                function_responses = []
                
                for function_call in function_calls:
                    func_name = function_call.name
                    func_args = dict(function_call.args)

                    # Execute the function
                    result = self.py_funcs[func_name](**func_args)

                    # Create function response
                    function_response = genai.protos.Part(
                        function_response=genai.protos.FunctionResponse(
                            name=func_name,
                            response={"result": result}
                        )
                    )
                    function_responses.append(function_response)

                # Send all function responses back to model
                response = chat.send_message(function_responses)
                
            else:
                # No more function calls, return final response
                break
        
        return response.text, chat.history
    
    def get_current_list(self) -> List[str]:
        """Get current shopping list."""
        return self.shopping_list.copy()
    
    def clear_list(self) -> None:
        """Clear the shopping list."""
        self.shopping_list = []
        self._save_shopping_list()


# Convenience function for quick access
def create_shopping_agent(shopping_list_file: str = None, api_key: str = None) -> ShoppingListAgent:
    """
    Create a new shopping list agent.
    
    Args:
        shopping_list_file: Path to shopping list file (optional)
        api_key: Google API key (optional, uses env var if not provided)
        
    Returns:
        ShoppingListAgent instance
    """
    return ShoppingListAgent(shopping_list_file, api_key)


# Example usage and testing
if __name__ == "__main__":
    # Demo the agent
    agent = create_shopping_agent()
    
    print("Smart Shopping List Agent Demo")
    print("=" * 50)
    
    # Test with some ingredients
    ingredients = ["pasta", "tomato sauce", "ground beef", "parmesan cheese", "garlic bread"]
    
    print(f"\nProcessing ingredients: {ingredients}")
    response, _ = agent.process_ingredients(ingredients)
    print(f"Agent response: {response}")

    print(f"\nCurrent shopping list: {agent.get_current_list()}")