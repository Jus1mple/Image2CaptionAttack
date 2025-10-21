# -*- coding:utf-8 -*-
# Judgment Template

prompt_template_v1 = """
        Please act as an impartial judge and evaluate the quality and similarity of the generated response provided by an AI assistant compared to the ground truth responses below. You will score the generated response on a scale of 1 to 5 based on its fluency, coherence, and relevance to the ground truth responses. Ensure that your evaluation is objective, avoiding any position biases, and that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation.

        Scoring Criteria:
        - Fluency: How natural and grammatically correct is the generated response?
        - Coherence: How logically consistent and clear is the generated response in relation to the context or ground truth?
        - Relevance: How closely does the generated response align with the content and meaning of the ground truth responses?

        Scoring Scale:
        - 1: Poor fluency, incoherent or irrelevant to the ground truth responses.
        - 2: Limited fluency, somewhat coherent, but the relevance to the ground truth is low.
        - 3: Moderate fluency, fairly coherent, with some relevance to the ground truth.
        - 4: Good fluency, mostly coherent, and relevant to the ground truth.
        - 5: Excellent fluency, very coherent, and highly relevant to the ground truth.

        Please provide detailed feedback on the reasons for your score. You may refer to the following examples for your reference:

        ---

        ### Example 1:

        [Generated]: "A bunch of bananas sitting on a table."  
        [Ground Truth List]: 
        - "A store display filled with ripe, unripe bananas and other fruit."  
        - "A group of bananas surround a small display of kiwi."  
        - "A fruit stand with plantains, kiwis, and bananas."  
        - "A fruit stand that has bananas, papaya, and plantains."  
        - "A fruit stand display with bananas and kiwi."

        Fluency: 5  
        The response is grammatically correct and reads naturally.  

        Coherence: 5  
        The sentence logically makes sense and is consistent with the context of bananas on a table.  

        Relevance: 5  
        The response is highly relevant to the list of ground truth responses, which focus on bananas and their display.  

        Final Score: 5  
        The generated response is fluent, coherent, and highly relevant to the ground truth.

        ---

        ### Example 2:

        [Generated]:" A table with bananas."  
        [Ground Truth List]:  
        - "A store display filled with ripe, unripe bananas and other fruit."  
        - "A group of bananas surround a small display of kiwi."  
        - "A fruit stand with plantains, kiwis, and bananas."  
        - "A fruit stand that has bananas, papaya, and plantains."  
        - "A fruit stand display with bananas and kiwi."

        Fluency: 5  
        The sentence is clear, concise, and grammatically correct.  

        Coherence: 4  
        The response is coherent but lacks additional context that might make it more specific, like the surrounding items in the display.  

        Relevance: 4  
        The response is relevant but does not include the same level of detail found in the ground truth, which includes more specific descriptions of the fruit stand and surrounding items.  

        Final Score: 4  
        A good response, but it lacks some detail and specificity compared to the ground truth.

        ---

        ### Example 3:

        [Generated]: "A fruit bowl with a bunch of bananas and some apples."  
        [Ground Truth List]:  
        - "A store display filled with ripe, unripe bananas and other fruit."  
        - "A group of bananas surround a small display of kiwi."  
        - "A fruit stand with plantains, kiwis, and bananas."  
        - "A fruit stand that has bananas, papaya, and plantains."  
        - "A fruit stand display with bananas and kiwi."

        Fluency: 5  
        The sentence is fluent and grammatically correct.  

        **Coherence:** 4  
        The response is coherent, but introducing apples creates a slight deviation from the ground truth context, which mostly focuses on bananas.  

        **Relevance:** 3  
        While bananas are relevant, the inclusion of apples introduces a mismatch with the ground truth, making the response less relevant overall.  

        **Final Score:** 4  
        A generally good response, but the introduction of apples reduces relevance.

        ---

        ### Example 4:

        [Generated]: "There are bananas on the fruit stand."  
        [Ground Truth List]:  
        - "A store display filled with ripe, unripe bananas and other fruit."  
        - "A group of bananas surround a small display of kiwi."  
        - "A fruit stand with plantains, kiwis, and bananas."  
        - "A fruit stand that has bananas, papaya, and plantains."  
        - "A fruit stand display with bananas and kiwi."

        Fluency: 4  
        The sentence is grammatically correct, but slightly vague in terms of detail.  

        Coherence: 4  
        The response is coherent, but it could benefit from more detail about the specific contents of the fruit stand, as in the ground truth responses.  

        Relevance: 3  
        The sentence is somewhat relevant but lacks the detail and specificity found in the ground truth responses, which describe more than just the bananas.  

        Final Score: 4  
        Clear and coherent, but lacking detail compared to the ground truth.

        ---

        ### Example 5:

        [Generated]: "There is a dog in the picture."  
        [Ground Truth List]: 
        - "A store display filled with ripe, unripe bananas and other fruit."  
        - "A group of bananas surround a small display of kiwi."  
        - "A fruit stand with plantains, kiwis, and bananas."  
        - "A fruit stand that has bananas, papaya, and plantains."  
        - "A fruit stand display with bananas and kiwi."

        Fluency: 5  
        The sentence is grammatically correct and clear.  

        Coherence: 1  
        The response is incoherent with the context of the ground truth, as it introduces an unrelated subject (dog).  

        Relevance: 1  
        The response is completely irrelevant to the ground truth, which describes scenes with bananas and fruit, not a dog.  

        Final Score: 1  
        The generated response is irrelevant and inconsistent with the given context.

        ---
        
        Based on the above examples, please evaluate the generated response in relation to the ground truth responses provided. Score the generated response on a scale of 1 to 5 from the perspective of fluency, coherence, and relevance to the ground truth responses. And give the final score based on the average of these three criteria. Provide detailed feedback on the reasons for your score.
        Now, please evaluate the following response:
        [Generated]: "{}"
        [Ground Truth List]: {}
    """



def get_prompt_from_template(generated_caption, ground_truth_list, template):
    ground_truth_list_str = "\n".join([f"- \"{gt}\"" for gt in ground_truth_list])
    return template.format(generated_caption, ground_truth_list_str)