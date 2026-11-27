import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html')
        own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}; own[404]=4
        base={'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':[404,19,31,15,84],'stone':1200}
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(base)); await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400); await pg.click('#tab-arena'); await pg.wait_for_timeout(300)
        code=await pg.inner_text('#mycode'); print('code',code)
        await pg.fill('#fcode',code); await pg.click('#ahelp-btn'); await pg.wait_for_timeout(300); print('self:',await pg.evaluate("JSON.parse(localStorage.getItem('teawase_v1')).hlp||null"))
        base['party']=[19,31,15,84,20]; await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(base)); await pg.reload(); await pg.wait_for_timeout(500)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400); await pg.click('#tab-arena'); await pg.wait_for_timeout(300)
        await pg.fill('#fcode',code); await pg.click('#ahelp-btn'); await pg.wait_for_timeout(400); await pg.screenshot(path='/tmp/h_1.png')
        print(await pg.evaluate("JSON.stringify(JSON.parse(localStorage.getItem('teawase_v1')).hlp)"))
        await pg.click('#tab-home'); await pg.wait_for_timeout(200); await pg.click('#go-btn')
        for _ in range(40):
            if await pg.query_selector('#anime:not([hidden])'): await pg.click('#anime')
            if await pg.query_selector('.pick:visible'): break
            await pg.wait_for_timeout(250)
        await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/h_2.png')
        print('picks',await pg.evaluate("[...document.querySelectorAll('.pick')].map(e=>e.id+':'+e.querySelector('.pw').innerText.trim())"))
        ov=await pg.evaluate("(()=>{const s=document.querySelector('#stage').getBoundingClientRect(),c=document.querySelector('.picks').getBoundingClientRect();return [Math.round(s.height),Math.round(c.bottom),innerHeight]})()"); print(ov)
        await pg.click('#pick-404'); await pg.wait_for_timeout(600); await pg.screenshot(path='/tmp/h_3.png')
        print(errs); await b.close()
asyncio.run(main())
