    
import http.client
import urllib
from agents.deals import Opportunity
from agents.agent import Agent

DO_PUSH = True

class MessagingAgent(Agent):

    name = "Messaging Agent"
    color = Agent.WHITE

    def __init__(self):
        """
        Set up this object to do push notifications via Pushover
        """
        self.log("Messaging Agent is initializing")
        if DO_PUSH:
            # ✅ Your credentials
            # self.pushover_user = ""  # Your User Key
            # self.pushover_token = ""  # Your API Token
            self.log("Messaging Agent has initialized Pushover")

    def push(self, text):
        """
        Send a Push Notification using the Pushover API
        """
        self.log("Messaging Agent is sending a push notification")
        conn = http.client.HTTPSConnection("api.pushover.net:443")
        conn.request(
            "POST", "/1/messages.json",
            urllib.parse.urlencode({
                "token": self.pushover_token,
                "user": self.pushover_user,
                "message": text,
                "sound": "cashregister"
            }),
            { "Content-type": "application/x-www-form-urlencoded" }
        )
        response = conn.getresponse()
        self.log(f"Pushover Response: {response.status} {response.reason}")

    def alert(self, opportunity: Opportunity):
        """
        Make an alert about the specified Opportunity
        """
        text = (
            f"Deal Alert! Price=${opportunity.deal.price:.2f}, "
            f"Estimate=${opportunity.estimate:.2f}, "
            f"Discount=${opportunity.discount:.2f} :"
            f"{opportunity.deal.product_description[:10]}... "
            f"{opportunity.deal.url}"
        )

        if DO_PUSH:
            self.push(text)

        self.log("Messaging Agent has completed")


# # ✅ Test without Opportunity object
# if __name__ == "__main__":
#     agent = MessagingAgent()
#     agent.push("Hello Hammad! ✅ Your Pushover notification is working!")
        