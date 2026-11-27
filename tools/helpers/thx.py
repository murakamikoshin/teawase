import asyncio,json,re
from playwright.async_api import async_playwright
own={i:2 for i in list(range(1,21))+[31,101,84,68,404]}
def base(party,extra=None):
    d={'owned':own,'enc':40,'joined':30,'mute':True,'tut':True,'tut2':True,'st0':1,'tgDone':True,'login':{'d':99991231,'n':1},'area':'forest','kira':{},'fav':84,'party':party,'stone':100}
    d.update(extra or {}); return d
async def openp(b,st,errs):
    ctx=await b.new_context(viewport={'width':390,'height':844}); pg=await ctx.new_page(); pg.on('pageerror',lambda e:errs.append(str(e)))
    await pg.add_init_script("window.__sh='';navigator.share=function(o){window.__sh=(o&&o.text)||'';return Promise.resolve();};try{navigator.clipboard.writeText=function(t){window.__sh=t;return Promise.resolve();};}catch(e){}")
    await pg.goto('file:///home/claude/teawase/index.html'); await pg.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(st)); await pg.reload(); await pg.wait_for_timeout(500)
    await pg.click('#title .t-tap', force=True); await pg.wait_for_timeout(400); return pg
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); errs=[]
        O=await openp(b,base([404,19,31,15,84]),errs)            # 呼ばれる 側(持ち主)
        await O.screenshot(path='/tmp/thx_home.png')
        await O.click('#multi-btn'); await O.wait_for_timeout(400); print('multi → page', await O.evaluate("!!document.querySelector('#raid-btn')"))
        await O.click('#tab-arena'); await O.evaluate("0"); 
        await O.click('[data-act=pg][data-id="-1"]:visible'); await O.wait_for_timeout(300)
        code=await O.inner_text('#mycode')
        C=await openp(b,base([19,31,15,84,20]),errs)             # 呼ぶ 側
        await C.click('#tab-arena'); await C.wait_for_timeout(300); await C.fill('#fcode',code); await C.click('#ahelp-btn'); await C.wait_for_timeout(400)
        print('btn before help used:', await C.evaluate("!!document.querySelector('#hthx-btn')"))
        st=json.loads(await C.evaluate("localStorage.getItem('teawase_v1')")); st['hlp']['n']=2
        await C.evaluate("v=>localStorage.setItem('teawase_v1',v)",json.dumps(st)); await C.reload(); await C.wait_for_timeout(500); await C.click('#title .t-tap', force=True); await C.wait_for_timeout(300); await C.click('#tab-arena'); await C.wait_for_timeout(300)
        await C.click('#hthx-btn'); await C.wait_for_timeout(400); txt=await C.evaluate("window.__sh"); m=re.search(r'HP-[0-9A-Z]+',txt or ''); print('share:',(txt or '')[:60].replace('\n',' '), m and m.group(0))
        await C.screenshot(path='/tmp/thx_c.png')
        hp=m.group(0)
        await O.click('[data-act=pg][data-id="1"]:visible'); await O.wait_for_timeout(200); await O.click('[data-act=pg][data-id="1"]:visible'); await O.wait_for_timeout(300)
        await O.click('[data-act=pg][data-id="-1"]:visible'); await O.wait_for_timeout(200); await O.click('[data-act=pg][data-id="-1"]:visible'); await O.wait_for_timeout(200)
        await O.fill('#fcode',hp); await O.click('#afriend-btn'); await O.wait_for_timeout(400); print('via friend field:', json.loads(await O.evaluate("localStorage.getItem('teawase_v1')")).get('shard'), await O.evaluate("(document.querySelector('.notice')||{}).textContent"))
        await O.click('[data-act=pg][data-id="1"]:visible'); await O.wait_for_timeout(200); await O.click('[data-act=pg][data-id="1"]:visible'); await O.wait_for_timeout(300)
        for k in range(2):
            await O.fill('#icode',hp); await O.click('#invuse-btn'); await O.wait_for_timeout(400)
            st=json.loads(await O.evaluate("localStorage.getItem('teawase_v1')")); print('use',k,st.get('shard'),st.get('hthx'), await O.evaluate("(document.querySelector('.notice')||{}).textContent"))
        await C.click('[data-act=pg][data-id="1"]:visible'); await C.wait_for_timeout(200); await C.click('[data-act=pg][data-id="1"]:visible'); await C.wait_for_timeout(300)
        await C.fill('#icode',hp); await C.click('#invuse-btn'); await C.wait_for_timeout(300); print('self:', await C.evaluate("(document.querySelector('.notice')||{}).textContent"))
        await C.fill('#icode',hp[:-1]+('A' if hp[-1]!='A' else 'B')); await C.click('#invuse-btn'); await C.wait_for_timeout(300); print('bad:', await C.evaluate("(document.querySelector('.notice')||{}).textContent"))
        print(errs); await b.close()
asyncio.run(main())
