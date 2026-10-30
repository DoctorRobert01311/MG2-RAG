# 第一天：Git、GitHub 与 VS Code

本仓库用于学习 MG²-RAG，保留作者代码、许可证和提交历史。
作者仓库：https://github.com/Daboolu/MG2-RAG
学习起点提交：91f0eed6fc5f4383f0bae746d12850f05e381e77

## 三个工具

- Git：记录本地代码修改历史。
- GitHub：保存远程仓库、协作和同步。
- VS Code：查看和修改本地文件，通过终端或源代码管理使用 Git。

## 当前结构

- upstream：作者的远程仓库。
- origin：待 GitHub Fork 创建成功后配置为自己的仓库。
- main：保留下载时的原始主线。
- study/day01-git：我们的学习分支。

## 在 VS Code 练习

1. 左侧资源管理器打开 README.md、main.py 和本文件。
2. 点击“终端 → 新建终端”，运行下面的只读命令。
3. 打开左侧“源代码管理”，观察修改与提交历史。

```bash
git status
git remote -v
git branch
git log --oneline -5
```

修改本文件，在下面写一句自己的理解，再保存。用 git diff 查看修改，然后仅提交本文件：

```bash
git diff -- study-notes/01-git-vscode.md
git add study-notes/01-git-vscode.md
git commit -m "docs: record my understanding of Git workflow"
```

只有 origin 配置完成且登录可用后，才推送：

```bash
git push -u origin study/day01-git
```

我的理解：Git是一种文件管理工具，我们通过相应的指令，将文件扒到线上仓库，然后再进行相应的记录，这里每一条记录都是会实时显示我们做的更正
          我的理解：分支共享之前的提交历史，之后可以独立修改和提交。

注意：不要提交密钥、.env、模型权重或下载的数据。今天先完成 Git 工作流，不需要运行 main.py 或安装 CUDA。


/Users/aligncsphd/Documents/Codex/2026-10-29/referenced-chatgpt-conversation-this-is-an/outputs/MG2-RAG/study-notes/01-git-vscode.md