import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[84,19,31,15,404],'stone':3000,'medal':5}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(500)
        await pg.click('#tab-shop'); await pg.wait_for_timeout(300)
        await pg.evaluate("document.querySelector('.shopx').scrollTop=99999"); await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/q_shop2.png')
        await pg.evaluate("document.querySelector('.shopx').scrollTop=700"); await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/q_shop1.png')
        await pg.click('#tab-dex'); await pg.wait_for_timeout(300)
        await pg.evaluate("showCardT&&0").catch(lambda e:0) if False else None
        await pg.click('#tab-gacha'); await pg.wait_for_timeout(300); await pg.click('#g-pull10'); 
        for i in range(60):
            await pg.mouse.click(195,420); await pg.wait_for_timeout(350)
            if await pg.evaluate("!!document.querySelector('.pe-cell')"): break
        await pg.wait_for_timeout(1200); await pg.screenshot(path='/tmp/q_res.png')
        await pg.click('[data-g=back]'); await pg.wait_for_timeout(400); await pg.click('#tab-dex'); await pg.wait_for_timeout(300)
        await pg.click('.dx-cell, .cell, [data-card]'); await pg.wait_for_timeout(600); await pg.screenshot(path='/tmp/q_card.png')
        print(errs); await b.close()
asyncio.run(main())
