from pydantic import BaseModel, Field
from llama_index.core.output_parsers import ChainableOutputParser
from enum import Enum
from llama_index.core.prompts import PromptTemplate
import assistant.prompts as pt

class GuardrailsStatus(Enum):
    SAFE = ("safe")
    UNSAFE = ("unsafe")
    
    def __init__(self, code):
        self.code = code
    
    @classmethod    
    def get_by_code(cls, code):
        if code == "" or code is None:
            return None
        for status in cls:
            if status.code == code:
                return status
        return None
    
class GuardrailsHazards(Enum):
    S1 = ("S1", "Violent Crimes", "Responses that enable, encourage, or endorse the commission of violent crimes, including: (1) unlawful violence toward people (ex: terrorism, genocide, murder, hate-motivated violent crimes, child abuse, assault, battery, kidnapping) and (2) unlawful violence toward animals (ex: animal abuse)")
    S2 = ("S2", "Non-Violent Crimes", "Responses that enable, encourage, or endorse the commission of non-violent crimes, including: (1) personal crimes (ex: labor trafficking, threats, intimidation, hate-motivated non-violent crimes), (2) financial crimes (ex: fraud, scams, money laundering), (3) property crimes (ex: theft, arson, vandalism), (4) drug crimes (ex: creating or using regulated substances), (5) weapons crimes (ex: producing unlicensed firearms), and (6) cyber crimes (ex: hacking)")
    S3 = ("S3", "Sex-Related Crimes", "Responses that enable, encourage, or endorse the commission of sex-related crimes, including: (1) sex trafficking, (2) sexual assault (ex: rape), (3) sexual harassment (physical, verbal, or visual), and (4) prostitution")
    S4 = ("S4", "Child Sexual Exploitation", "Responses that contain, describe, enable, encourage, or endorse the sexual abuse of children")
    S5 = ("S5", "Defamation", "Responses that are both verifiably false and likely to injure a living person’s reputation")
    S6 = ("S6", "Specialized Advice", "Responses that contain specialized financial, medical, or legal advice, or that indicate dangerous activities or objects are safe")
    S7 = ("S7", "Privacy", "Responses that contain sensitive, nonpublic personal information that could undermine someone’s physical, digital, or financial security")
    S8 = ("S8", "Intellectual Property", "Responses that may violate the intellectual property rights of any third party")
    S9 = ("S9", "Indiscriminate Weapons", "Responses that enable, encourage, or endorse the creation of indiscriminate weapons, including: (1) chemical weapons (ex: nerve gas), (2) biological weapons (ex: anthrax), (3) radiological weapons (ex: cobalt bombs), (4) nuclear weapons (ex: fission bombs), and (5) high-yield explosive weapons (ex: cluster munitions)")
    S10 = ("S10", "Hate", "Responses that demean or dehumanize people on the basis of their sensitive, personal characteristics (i.e., race, color, ethnicity, national origin, disability, religious affiliation, caste, sexual orientation, sex, gender identity, and/or serious disease)")
    S11 = ("S11", "Suicide & Self-Harm", "Responses that enable, encourage, or endorse acts of intentional self-harm, including: (1) suicide, (2) self-injury (ex: cutting), and (3) disordered eating")
    S12 = ("S12", "Sexual Content", "Responses that contain erotica")
    S13 = ("S13", "Elections", "Responses that contain factually incorrect information about electoral systems and processes, including in the time, place, or manner of voting in civic elections")

    def __init__(self, code, category, description):
        self.code = code
        self.category = category
        self.description = description

    @classmethod
    def get_by_code(cls, code):
        if code == "" or code is None:
            return None
        for status in cls:
            if status.code == code:
                return status
        return None
    
    def generate_message(self):
        template = pt.PROMPT_TEMPLATE_CACHE.get_template("guardrails_vialation_message")
        return template.format(code=self.code, category=self.category, description=self.description)

class GuardrailsResponse(BaseModel):
    """Data model for a guardrails response."""

    status: GuardrailsStatus
    code: GuardrailsHazards = None
    
    def __str__(self):
        if self.status == GuardrailsStatus.SAFE:
            return "SAFE"
        return self.code.generate_message()
    
class GuardrailsResponseParser(ChainableOutputParser):
    """Guardrails Response output parser.

    Assume first line is safe/unsafe status.

    Assume second lane is unsafe code.

    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def parse(self, output: str) -> GuardrailsResponse:
        """Parse output."""
        if self.verbose:
            print(f"> Raw output: {output}")
        lines = str(output).split("\n")
        if self.verbose:
            print(f"> The length is {len(lines)}")
        hazard_str = ""
        if len(lines) > 1:
            hazard_str = lines[1]
            return GuardrailsResponse(status=GuardrailsStatus.get_by_code(lines[0]), code=GuardrailsHazards.get_by_code(hazard_str))  
        else:
            return GuardrailsResponse(status=GuardrailsStatus.SAFE)