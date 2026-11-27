import asyncio,json,time
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:1 for i in list(range(1,21))}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':15,'party':[15,19,31]}))
        await pg.reload(); await pg.wait_for_timeout(500); await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(300)
        for name,rs in [('全部★3',[3]*10+[4]),('ふつう',[3,3,4,3,3,3,4,3,3,3,4]),('当たり多め',[3,4,5,3,3,4,3,6,3,3,4])]:
            fo=[{'a':(0 if r<=3 else r-3)} for r in rs]
            await pg.evaluate("window.__d=0;window.__t0=performance.now()")
            await pg.evaluate("([rs,fo])=>window.__geRun(rs,fo,function(){window.__d=performance.now()-window.__t0;})",[rs,fo])
            for _ in range(120):
                await pg.wait_for_timeout(500)
                d=await pg.evaluate("window.__d")
                if d: break
            print(name, round(d/1000,1),'秒')
        print(errs); await b.close()
asyncio.run(main())
