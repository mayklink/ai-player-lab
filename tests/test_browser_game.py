import asyncio
from ai_player_lab.browser_game import BrowserGame


def test_release_deadline_does_not_wait_for_next_capture():
    class Keys:
        def __init__(self):self.events=[]
        async def down(self,key):self.events.append(('down',key))
        async def up(self,key):self.events.append(('up',key))
    class Page:
        keyboard=Keys()
    async def scenario():
        game=BrowserGame('https://example.test/game');game.page=Page()
        await game.set_keys({'ArrowRight'},30)
        # A slow screenshot/network operation must not hold the button for its duration.
        await asyncio.sleep(.09)
        assert not game.held
        assert game.page.keyboard.events==[('down','ArrowRight'),('up','ArrowRight')]
    asyncio.run(scenario())


def test_rejects_browser_shortcuts_before_dispatch():
    async def scenario():
        game=BrowserGame('https://example.test/game')
        try:await game.set_keys({'Control'})
        except ValueError:return
        raise AssertionError('Browser shortcut accepted')
    asyncio.run(scenario())
