from dotenv import load_dotenv
load_dotenv()   # must be FIRST

import os
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

print("---- ENV CHECK ----")
for k, v in os.environ.items():
    if "LANG" in k or "SMITH" in k:
        print(k, "=", v)
print("-------------------")

print("TRACING:", os.getenv("LANGSMITH_TRACING"))
print("API:", os.getenv("LANGSMITH_API_KEY")[:10])
print("PROJECT:", os.getenv("LANGSMITH_PROJECT"))

def main():
    print("Hello from langchain-course!")
    information = """
    Cristiano Ronaldo dos Santos Aveiro[a] (born 5 February 1985), nicknamed CR7, is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al-Nassr and the Portugal national team. Widely regarded as one of the greatest players in history, he has won numerous individual accolades throughout his career, including five Ballon d'Ors, a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes. He was also named the world's best player five times by FIFA.[note 3] He has won 34 trophies in his career, including five UEFA Champions Leagues and the UEFA European Championship. He holds the records for most goals (140) and assists (42) in the Champions League, goals (14) and assists (8) in the European Championship, and most international appearances (226) and international goals (143). He is the only player to have scored 100 goals with four different clubs. He has made over 1,300 professional career appearances, the most by an outfield player, and has scored over 960 official senior career goals for club and country, making him the top goalscorer of all time.

    Born in Funchal, Madeira, Ronaldo began his career with Sporting CP before signing with Manchester United in 2003. He became a star player at United, where he won three consecutive Premier League titles, the Champions League, and the FIFA Club World Cup. His 2007–08 season earned him his first Ballon d'Or at age 23. In 2009, Ronaldo became the subject of the then-most expensive transfer in history when he joined Real Madrid in a deal worth €94 million (£80 million). At Madrid, he was at the forefront of the club's resurgence as a dominant European force, helping them win four Champions Leagues between 2014 and 2018, including the long-awaited La Décima. He also won two La Liga titles, including the record-breaking 2011–12 season in which Madrid reached 100 points, and became the club's all-time top goalscorer. He won Ballon d'Ors in 2013, 2014, 2016 and 2017, and was runner-up three times to Lionel Messi, his perceived career rival. Following issues with the club hierarchy, Ronaldo signed for Juventus in 2018 in a transfer worth an initial €100 million, where he was pivotal in winning two Serie A titles. In 2021, he returned to United before joining Al-Nassr in 2023.
        """
    
    summary_template = """
    given the information {information} about a person I want you to create:
    1. A short summary
    2. His achievements list
    """

    summary_prompt_template = PromptTemplate(
        input_variables = ["information"], template = summary_template
    )

    llm = ChatOllama(temperature=0, model = "gemma3:270m")
    chain = summary_prompt_template | llm

    response = chain.invoke({"information": information})
    print(response.content)

if __name__ == "__main__":
    main()

