"""
LLM-powered rule optimization processor.
"""

# Import built-in modules
from typing import Any, Dict

# Import third-party modules
import openai

# Import local modules
from ai_rules.processors import RuleProcessor


class LLMOptimizeProcessor(RuleProcessor):
    """Processor for optimizing prompts using LLM."""
    
    def process(self, content: str, options: Dict[str, Any]) -> str:
        """Process the rule content using LLM optimization.
        
        Args:
            content: The rule content to optimize
            options: Processing options including:
                - model: LLM model to use
                - temperature: Sampling temperature
                
        Returns:
            Optimized content
        """
        model = options.get('model', 'gpt-4')
        temperature = options.get('temperature', 0.7)
        
        try:
            # Create optimization prompt
            prompt = f"""
            Please optimize the following AI assistant rules to make them more effective:

            {content}

            Please maintain the original structure but improve:
            1. Clarity and precision
            2. Completeness of instructions
            3. Error handling guidance
            4. Task organization
            """
            
            # Call OpenAI API
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are an expert in AI prompt engineering."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature
            )
            
            # Extract optimized content
            optimized_content = response.choices[0].message.content
            return optimized_content
            
        except Exception as e:
            raise Exception(f"LLM optimization failed: {str(e)}")
    
    def validate(self, content: str) -> bool:
        """Validate if the content can be optimized.
        
        Args:
            content: The content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        # Basic validation: check if content is not empty and is string
        if not isinstance(content, str) or not content.strip():
            return False
        return True
