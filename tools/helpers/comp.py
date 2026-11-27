import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(args=['--use-gl=swiftshader','--enable-unsafe-swiftshader'])
        pg=await b.new_page(viewport={'width':390,'height':844})
        errs=[]; pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html'); await pg.wait_for_timeout(500)
        await pg.evaluate("localStorage.clear()"); await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title', force=True); await pg.wait_for_timeout(300); (await pg.query_selector('#title:not([hidden])')) and await pg.click('#title', force=True); await pg.wait_for_timeout(400)
        await pg.screenshot(path='/tmp/c0.png')
        for i in range(4):
            if await pg.query_selector('#tut-next'): await pg.click('#tut-next'); await pg.wait_for_timeout(200)
        print('tut done:', json.loads(await pg.evaluate("localStorage.getItem('teawase_v1')")).get('tut'))
        own={i:1 for i in list(range(1,21))+[31,101,58,61]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'login':{'d':99991231,'n':1},'area':'forest','pity':12,'kira':{'61':True},'fav':61,'party':[61,58,15],'tut':True}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title', force=True); await pg.wait_for_timeout(300); (await pg.query_selector('#title:not([hidden])')) and await pg.click('#title', force=True); await pg.wait_for_timeout(500)
        await pg.screenshot(path='/tmp/c1.png')
        await pg.click('#showcase .np'); await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/c2.png')
        await pg.click('#share-btn'); await pg.wait_for_timeout(900); await pg.screenshot(path='/tmp/c3.png')
        src=await pg.evaluate("(document.querySelector('#card img')||{}).src||''")
        import base64
        if src.startswith('data:image/png'): open('/tmp/card.png','wb').write(base64.b64decode(src.split(',')[1])); print('card image saved')
        await pg.click('#card-close'); await pg.click('#tab-guide'); await pg.wait_for_timeout(200)
        for _ in range(4): await pg.click('#pg-next'); await pg.wait_for_timeout(100)
        await pg.screenshot(path='/tmp/c4.png')
        print('errors',errs); await b.close()
asyncio.run(main())
