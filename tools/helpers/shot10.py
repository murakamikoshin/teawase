import asyncio,json
from playwright.async_api import async_playwright
async def hit(pg):
    el=await pg.wait_for_selector('#hit-btn'); bx=await el.bounding_box()
    x=bx['x']+bx['width']/2-40; y=bx['y']+bx['height']/2
    await pg.mouse.move(x,y); await pg.mouse.down(); await pg.wait_for_timeout(120); await pg.mouse.move(x+80,y,steps=3); await pg.mouse.up()
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch()
        pg=await b.new_page(viewport={'width':390,'height':844})
        errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:1 for i in list(range(1,21))+[31,101]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'login':{'d':99991231,'n':1},'area':'forest','pity':12,'kira':{},'fav':15,'party':[15,19,31]}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
        await pg.screenshot(path='/tmp/q1.png')
        await pg.click('#go-btn'); await pg.wait_for_timeout(250); await pg.screenshot(path='/tmp/q2.png')
        async def clear():
            for _ in range(40):
                if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
                if await pg.query_selector('.pick:visible') or await pg.query_selector('#invite-btn'): return
                await pg.wait_for_timeout(250)
        await clear()
        await pg.screenshot(path='/tmp/q3.png')
        await pg.click('.pick'); await pg.wait_for_timeout(200); await pg.screenshot(path='/tmp/q4.png')
        await hit(pg); await pg.wait_for_timeout(330); await pg.screenshot(path='/tmp/q5.png')
        depth=0
        for rnd in range(4):
            for _ in range(9):
                await clear()
                if await pg.query_selector('#invite-btn'): break
                await pg.click('.pick'); await hit(pg); await pg.wait_for_timeout(1700)
            await pg.wait_for_selector('#invite-btn',timeout=9000)
            await pg.click('#invite-btn'); await pg.wait_for_timeout(1300)
            if rnd==0: await pg.screenshot(path='/tmp/q6.png')
            await pg.wait_for_timeout(1500)
            if rnd==0: await pg.screenshot(path='/tmp/q7.png')
            txt=await pg.inner_text('#again-btn'); print(rnd, txt.replace('\n',' '))
            await pg.click('#again-btn'); await pg.wait_for_timeout(400)
        await clear(); await pg.screenshot(path='/tmp/q8.png')
        print('errors',errs)
        await b.close()
asyncio.run(main())
