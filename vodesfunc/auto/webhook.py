"""
    WIP, kinda demotivated to do this shit
"""

from datetime import datetime
from time import sleep

from discord_webhook import DiscordEmbed, DiscordWebhook
from StringProgressBar import progressBar


__all__: list[str] = [
    'Webhook',
]


class Webhook:

    sent = None
    url: str
    show_name: str
    episode: str
    last_updated: datetime
    prev_progress: int = 0
    discord_webhook: DiscordWebhook

    def __init__(self, url: str, show_name: str, episode: str) -> None:
        self.url = url
        self.show_name = show_name
        self.episode = episode
        discord_webhook = DiscordWebhook(self.url, content="Initializing...")
        sleep(10)
        self.sent = discord_webhook.execute()

    def update_message(self, process: str = 'Encode', details: str = "", fields: dict = {}, progress: int = 0, total: int = 0) -> bool | None:
        now = datetime.now()
        if ((self.last_updated.second + 5) < now.second) or self.sent is None:
            return None

        bar = progressBar().filledBar(total=total, current=progress, size=80)

        self.discord_webhook.content = ""

        self.discord_webhook.remove_embeds()
        embed = DiscordEmbed(self.show_name + " - " + self.episode)
        embed.set_author(process + " running...")
        newline = "\n"
        padded_details = "\n" + details
        embed.set_description(f'Progress:```{newline}{bar[0]} {bar[1]}```{padded_details if details else ""}')
        for key, val in fields:
            embed.add_embed_field(str(key), str(val), True)
        self.discord_webhook.add_embed(embed)

        self.sent = self.discord_webhook.edit(self.sent)
        self.prev_progress = progress
        #requests.patch(self.url + "/messages/" + self.message_id)
        self.last_updated = now
