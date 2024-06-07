from easydict import EasyDict as edict

DirectPromptingTemplate = """
${question} 
Q: Do you need additional information to answer this question? 
A:
"""

template = edict({"DirectPromptingTemplate": DirectPromptingTemplate})



