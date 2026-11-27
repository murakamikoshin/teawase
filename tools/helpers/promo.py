import asyncio,json
from playwright.async_api import async_playwright
import cv2,numpy as np
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:1 for i in list(range(1,21))}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':15,'party':[15,19,31]}))
        await pg.reload(); await pg.wait_for_timeout(500); await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
        await pg.evaluate("window.__geRun([6],[{a:1}],function(){})")       # 紫 → 虹
        fr=[]
        for i in range(24):
            await pg.wait_for_timeout(350); await pg.screenshot(path='/tmp/_p.png'); fr.append(cv2.resize(cv2.imread('/tmp/_p.png'),(195,422)))
        cv2.imwrite('/tmp/promo.png',np.vstack([np.hstack(fr[k:k+8]) for k in range(0,24,8)])); print(errs); await b.close()
asyncio.run(main())
