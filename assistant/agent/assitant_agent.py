from typing import Any, List
from llama_index.core.workflow import (
    Event,
    StartEvent,
    StopEvent,
    Workflow,
    Context,
    step,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.llms.function_calling import FunctionCallingLLM
from llama_index.core.tools.types import BaseTool
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms.llm import LLM
from assistant.pydantic.guardrails import GuardrailsResponse, GuardrailsStatus, GuardrailsResponseParser
from llama_index.core.prompts import PromptTemplate
import assistant.prompts as pt
from assistant.pydantic.user_intention import UserIntent
from llama_index.program.guidance import GuidancePydanticProgram
from llama_index.core.program import LLMTextCompletionProgram
import random
from llama_index.core.base.llms.types import CompletionResponse

class ClearedRequestEvent(Event):
    request: str

class HarmfullRequestEvent(Event):
    request: str
    
class InformationRequestEvent(Event):
    request: str    
    
class RetrivalRequestEvent(Event):
    request: str
    
class RetrivalResultEvent(Event):
    result: str

class ClearedResultEvent(Event):
    result: str    
        
class FAQResultEvent(Event):
    result: str
    
class SearchRequestEvent(Event):
    request: str

class SearchResultEvent(Event):
    result: str    
        
class JiraTicketCreationRequestEvent(Event):
    request: str

class JiraTicketCreationResultEvent(Event):
    result: str
    
class AssystantAgent(Workflow):
    faq = {
        "Help", "I can "
    }
    
    def __init__(
        self,         
        llm: LLM | None = None,
        guardian_llm: LLM | None = None,
        tools: List[BaseTool] | None = None,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.tools = tools or []

        self.llm = llm or OpenAI()
        self.guardian_llm = guardian_llm or self.llm
        assert self.llm.metadata.is_function_calling_model

        self.memory = ChatMemoryBuffer.from_defaults(llm=llm)
        self.sources = []
        self.user_intent_template = pt.PROMPT_TEMPLATE_CACHE.get_template("user_intent_router")
        self.faq_template = pt.PROMPT_TEMPLATE_CACHE.get_template("faq")
        self.router_program = LLMTextCompletionProgram.from_defaults(
            output_cls=UserIntent,
            prompt=self.user_intent_template,
            guidance_llm=self.llm,
            verbose=True,
        )
        
    @step
    async def inbound_guardrails(self, ctx: Context, event: StartEvent) -> ClearedRequestEvent | StopEvent:
        request = event.get("request")
        prompt = f"{request}"
        response = await self.guardian_llm.acomplete(prompt=prompt)
        guardrails_response = GuardrailsResponseParser().parse(response)
        await ctx.set("inbound_guardrails_response", guardrails_response)
        if guardrails_response.status == GuardrailsStatus.UNSAFE:                         
            return StopEvent(result=guardrails_response)
        else:
            return ClearedRequestEvent(request=request)
    
    @step
    async def router(self, ctx: Context, event: ClearedRequestEvent) -> InformationRequestEvent | SearchRequestEvent | JiraTicketCreationRequestEvent:
        request = event.request        
        intent = self.router_program(user_input=request, verbose=True)
        await ctx.set("intent", intent)
        if intent.user_intent == "AUTODESK_SEARCH":
            return SearchRequestEvent(request=request)
        else:
            if intent.user_intent == "JIRA_TICKET_CREATION":
                return JiraTicketCreationRequestEvent(request=request)
        
        return InformationRequestEvent(request=request)
    
    @step
    async def jira_ticket_creation(self, ctx: Context, event: JiraTicketCreationRequestEvent) -> JiraTicketCreationResultEvent:
        intent = await ctx.get("intent")
        result= f"The jira Ticket #{random.randint(1, 999)} was created.\n User intent: {intent.user_intent}, the orignal request: {intent.user_input}"
        return JiraTicketCreationResultEvent(result=result)
    
    @step
    async def search_request(self, ctx: Context, event: SearchRequestEvent) -> SearchResultEvent:
        intent = await ctx.get("intent")
        result= f"Looking for answer '{event.request}' on AutoDesk site.\n User intent: {intent.user_intent}, the orignal request: {intent.user_input}"
        return SearchResultEvent(result=result)
    
    @step
    async def faq(self, event: InformationRequestEvent) -> FAQResultEvent | RetrivalRequestEvent :
        response = await self.llm.acomplete(prompt=self.faq_template.format(user_input=event.request), verbose=True)
        print(f"FAQ Response: {response.text}")        
        if "==NONE==" in response.text:
            return RetrivalRequestEvent(request=event.request)
        return FAQResultEvent(result=response.text)
    
    @step
    async def retrival(self, ctx: Context, event: RetrivalRequestEvent) -> RetrivalResultEvent :
        intent = await ctx.get("intent")
        result= f"Looking for answer '{event.request}' on the corporate knowledge base.\n User intent: {intent.user_intent}, the orignal request: {intent.user_input}"
        return RetrivalResultEvent(result=result)
    
    @step
    async def outbound_guardrails(self, ctx: Context, event: FAQResultEvent | RetrivalResultEvent | SearchResultEvent) -> ClearedResultEvent :
        result = event.result
        prompt = f"{result}"
        response = await self.guardian_llm.acomplete(prompt=prompt)
        guardrails_response = GuardrailsResponseParser().parse(response)
        await ctx.set("outbound_guardrails_response", guardrails_response)
        if guardrails_response.status == GuardrailsStatus.UNSAFE:                         
            return StopEvent(result=guardrails_response)
        else:
            return ClearedResultEvent(result=event.result)
    
    @step
    async def response(self, event: ClearedResultEvent | JiraTicketCreationResultEvent) -> StopEvent :
        
        return StopEvent(result=event.result)
    
async def main():
    
    w = AssystantAgent(llm=OpenAI(model="gpt-4o-mini"))
    result = await w.run(request="Hello")
    print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())