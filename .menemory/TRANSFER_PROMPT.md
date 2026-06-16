# MoNaVLA Transfer Prompt

Use the following content when you need to move this project context to another server or agent.

## Injection Prompt

> You are taking over an existing MoNaVLA research session. Before doing any planning or coding, read `.menemory/master_snapshot.md` and `.menemory/core/master_memory.md`, register their contents as the active project memory, and continue from the post-BBox decision point instead of reconsidering BBox-first approaches.

## Minimum Files To Copy

- `.menemory/master_snapshot.md`
- `.menemory/core/master_memory.md`

## Recommended First Command On The New Server

```bash
sed -n '1,220p' .menemory/master_snapshot.md
sed -n '1,240p' .menemory/core/master_memory.md
```

## Notes

- Update absolute paths if the repository lives somewhere other than `/home/soda/MoNaVLA`.
- Do not put API keys into transfer prompts or tracked markdown files.
