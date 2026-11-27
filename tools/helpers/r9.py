import asyncio,json
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(viewport={'width':390,'height':844}); errs=[]
        pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html'); await pg.evaluate("localStorage.clear()"); await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps({'owned':{'1':1},'enc':0,'joined':0,'mute':True}))
        await pg.reload(); await pg.wait_for_timeout(600)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(500)
        await pg.click('#tut-skip'); await pg.wait_for_timeout(400); await pg.screenshot(path='/tmp/n_1.png')
        await pg.click('#st0-2'); await pg.wait_for_timeout(300); await pg.screenshot(path='/tmp/n_2.png')
        await pg.click('#st0-ok'); await pg.wait_for_timeout(600); await pg.screenshot(path='/tmp/n_3.png')
        print(await pg.evaluate("localStorage.getItem('teawase_v1')"))
        st=json.loads(await pg.evaluate("localStorage.getItem('teawase_v1')")); st['enc']=1
        await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(st)); await pg.reload(); await pg.wait_for_timeout(600)
        await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(500); 
        if await pg.evaluate("!!document.querySelector('#card-close')&&!document.getElementById('card').hidden"): await pg.click('#card-close')
        await pg.wait_for_timeout(400); await pg.screenshot(path='/tmp/n_4.png')
        await pg.click('#tg-btn'); await pg.wait_for_timeout(500); await pg.screenshot(path='/tmp/n_5.png')
        await pg.click('#tg-pull',force=True)
        for i in range(70):
            await pg.mouse.click(195,420); await pg.wait_for_timeout(350)
            if await pg.evaluate("!!document.querySelector('#tg-keep')"): break
        await pg.wait_for_timeout(800); await pg.screenshot(path='/tmp/n_6.png')
        a=await pg.evaluate("JSON.parse(localStorage.getItem('teawase_v1')).tgRes.join()")
        await pg.click('#tg-re'); await pg.wait_for_timeout(600); await pg.screenshot(path='/tmp/n_7.png')
        print(a, await pg.evaluate("JSON.parse(localStorage.getItem('teawase_v1')).tgRes.join()"))
        await pg.click('#tg-keep'); 
        for i in range(40):
            await pg.mouse.click(195,300); await pg.wait_for_timeout(400)
            if await pg.evaluate("document.getElementById('gacha').hidden && !!document.querySelector('#dex-pedit')"): break
        await pg.wait_for_timeout(800); await pg.screenshot(path='/tmp/n_8.png')
        st=json.loads(await pg.evaluate("localStorage.getItem('teawase_v1')")); print(st['owned'],st['party'],st.get('tgDone'),st.get('gpity'))
        print(errs); await b.close()
asyncio.run(main())
