#!/bin/bash
# 作業場 → GitHub(murakamikoshin/teawase)に 差分を 送る。1回の push を 小さく する ため 40MB ごとに 分ける
set -e; R=/home/claude/repo_teawase; cd /home/claude
rm -rf $R/TeawaseApp; cp -r TeawaseApp $R/; cp -r tools/. $R/tools/; find $R/tools -name __pycache__ -prune -exec rm -rf {} +; cp /tmp/one.py /tmp/shot10.py /tmp/comp.py /tmp/sc.py $R/tools/helpers/ 2>/dev/null || true
cp -r artplan/. $R/artplan/; cp -r dots5/. $R/dots5/; cp -r newsrc/. $R/newsrc/; cp -r sheets/. $R/sheets/; cp hsgal/index.html $R/hsgal/; cp TODO_teawase.md DESIGN_renewal.md $R/; mkdir -p $R/assets_in $R/assets; cp -r assets_in/. $R/assets_in/; cp -r assets/. $R/assets/
cd $R; MSG="${1:-テアワセ！: 更新}"
commit(){ git commit -qm "$MSG

Co-Authored-By: Claude Fable 5.2 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01SZqiHnTXLBkqSoNbDEu828" && git push -q origin main 2>&1 | grep -v negotiat | tail -2; }
sz=0; n=0
git status --porcelain -uall | cut -c4- | while IFS= read -r f; do
  if [ -e "$f" ]; then s=$(stat -c %s "$f"); else s=0; fi
  if [ $((sz+s)) -gt 40000000 ] && [ $n -gt 0 ]; then commit; sz=0; n=0; fi
  git add -A -- "$f"; sz=$((sz+s)); n=$((n+1))
done
git add -A; git diff --cached --quiet || commit
git log --oneline | head -1; git status -sb | head -1
