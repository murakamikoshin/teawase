import asyncio,json,sys
from playwright.async_api import async_playwright
import cv2,numpy as np
W,H=(390,844) if len(sys.argv)<2 or sys.argv[1]!='s' else (375,667)
seq=sys.argv[2] if len(sys.argv)>2 else 'ten'
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':W,'height':H}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:1 for i in list(range(1,21))+[31,101]}; own.update({409:2,402:5})
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':15,'party':[15,19,31],'stone':99999,'medal':0,'gfree':99991231}))
        await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
        await pg.click('#tab-gacha'); await pg.wait_for_timeout(500)
        if seq=='ten':
            await pg.evaluate("window.__geForce=[{a:0},{a:0},{a:1},{a:0},{a:0},{a:1},{a:0},{a:0},{a:0},{a:0},{a:1}]")
            await pg.click('#g-pull10')
        else:
            rs=[int(x) for x in seq.split(',')]; fo=[{'a':max(0,(0 if r<=3 else r-3)-1)} for r in rs]
            if 'q' in sys.argv: await pg.evaluate("window.__q=1")
            await pg.evaluate("([rs,fo])=>window.__geRun(rs,fo,function(){})",[rs,fo])
        n=int(sys.argv[3]) if len(sys.argv)>3 else 30; dt=int(sys.argv[4]) if len(sys.argv)>4 else 300
        fr=[]
        for i in range(n):
            await pg.wait_for_timeout(dt); await pg.screenshot(path='/tmp/_f.png'); fr.append(cv2.resize(cv2.imread('/tmp/_f.png'),(W*240//390,H*240//390)))
        per=10; rows=[np.hstack(fr[k:k+per]+[np.zeros_like(fr[0])]*(per-len(fr[k:k+per]))) for k in range(0,len(fr),per)]
        cv2.imwrite('/tmp/gfilm.png',np.vstack(rows)); print(errs)
        await b.close()
asyncio.run(main())
