import asyncio,json,collections
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(); pg=await b.new_page(); errs=[]; pg.on('pageerror',lambda e:errs.append(str(e)))
        await pg.goto('file:///home/claude/teawase/index.html'); await pg.wait_for_timeout(800)
        a=await pg.evaluate("window.__skAudit()")
        print('chars',len(a),'no skill',[x['id'] for x in a if not x['sk']],'no trait',[x['id'] for x in a if not x['tr']])
        print('only-trait',[x['id'] for x in a if x['only']])
        print('星葬 names',[ (x['id'],x['sk'],x['tr']) for x in a if '星葬' in (x['sk'] or '')+(x['tr'] or '')])
        c=collections.Counter(x['sk'] for x in a); print('top skill names',c.most_common(8))
        for x in a:
            if x['id'] in (1,2,3,101,401,404,424,440,84,70,500): print(x['id'],x['n'],x['r'],'|',x['sk'],':',x['skd'],'|',x['tr'],':',x['trd'])
        print(errs); await b.close()
asyncio.run(main())
