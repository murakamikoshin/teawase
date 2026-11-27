import asyncio,json
from playwright.async_api import async_playwright
async def _safe(pg,sel,**kw):
    try:
        await pg.click(sel,force=True,timeout=2500,**kw); return True
    except Exception:
        return False
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[84,19,31,15,404],'stone':1200,'medal':5}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(500)
        await pg.screenshot(path='/tmp/r_home.png')
        tabs=['dex','arena','shop','guide']
        for t in tabs:
            await pg.click(f'#tab-{t}'); await pg.wait_for_timeout(350); await pg.screenshot(path=f'/tmp/r_{t}.png')
        await pg.click('#tab-home'); await pg.wait_for_timeout(200)
        await pg.click('#tab-gacha'); await pg.wait_for_timeout(400); await pg.screenshot(path='/tmp/r_gacha.png')
        await pg.click('#tab-home'); await pg.wait_for_timeout(200)
        await pg.click('#tab-dex'); await pg.click('#dex-pedit'); await pg.wait_for_timeout(400); await pg.screenshot(path='/tmp/r_party.png')
        await pg.click('#tab-home'); await pg.wait_for_timeout(200)
        await pg.click('#go-btn'); await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/r_go.png')
        for _ in range(40):
            if await pg.query_selector('#anime:not([hidden])'): await _safe(pg,'#anime')
            if await pg.query_selector('.pick:visible'): break
            await pg.wait_for_timeout(250)
        await pg.screenshot(path='/tmp/r_enc.png'); print(errs); await b.close()
asyncio.run(main())
