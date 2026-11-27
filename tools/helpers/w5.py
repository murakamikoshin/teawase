import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); errs=[]
        out=[]
        for w,h,hand in [(390,844,'scissors'),(375,667,'paper'),(390,844,'rock')]:
            pg=await b.new_page(viewport={'width':w,'height':h}); pg.on('pageerror',lambda e:errs.append(str(e)))
            await pg.goto('file:///home/claude/teawase/index.html')
            own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
            th={'rock':1,'scissors':1,'paper':1}; th.pop(hand)
            await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'tutH':th,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'field','kira':{},'fav':84,'party':[1,2,3,15,19]}))
            await pg.reload(); await pg.wait_for_timeout(500); await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
            await pg.click('#go-btn')
            for _ in range(40):
                if await pg.query_selector('#anime:not([hidden])'):
                    try: await pg.click('#anime',force=True,timeout=2000)
                    except Exception: pass
                if await pg.query_selector('.pick:visible'): break
                await pg.wait_for_timeout(250)
            pid={'rock':1,'scissors':2,'paper':3}[hand]
            await pg.click(f'#pick-{pid}',force=True); await pg.wait_for_timeout(700); await pg.screenshot(path=f'/tmp/w5_{hand}.png')
            print(hand, await pg.evaluate("!!document.querySelector('.tutfing')"), await pg.evaluate("(document.querySelector('.tutb')||{}).textContent"))
            await pg.close()
        print(errs); await b.close()
asyncio.run(main())
