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
        own={i:1 for i in list(range(1,21))+[31,101]}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':15,'party':[15,19,31],'stone':99999,'medal':50}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400)
        await pg.click('#tab-gacha'); await pg.wait_for_timeout(400)
        print('gacha open', bool(await pg.query_selector('#gacha:not([hidden])')))
        for sel in ('#g-pull','#g-pull10'):
            await pg.click(sel); await pg.wait_for_timeout(1300)
            print(sel,'egg', bool(await pg.query_selector('#gegg:not([hidden])')))
            for _ in range(80):
                if await pg.query_selector('#gegg:not([hidden])'): await _safe(pg,'#gegg',position={'x':100,'y':300})
                elif await pg.query_selector('#anime:not([hidden])'): await _safe(pg,'#anime')
                else: break
                await pg.wait_for_timeout(120)
            await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/g_'+sel[1:]+'.png')
            bk=await pg.query_selector('[data-g="back"]')
            if bk: await bk.click(); await pg.wait_for_timeout(200)
        print('errs',errs); await b.close()
asyncio.run(main())
