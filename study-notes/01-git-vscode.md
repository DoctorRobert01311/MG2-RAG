# Git 与 VS Code 实用笔记

## 我的理解

Git 是版本管理工具，可以在本地记录文件修改历史。分支共享之前的提交历史，之后可以独立修改和提交。GitHub 用于保存远程副本；使用 Git 不要求先上传 GitHub。

## 修改怎样成为一个版本

```text
工作区（编辑文件） → git add → 暂存区 → git commit → 本地提交历史 → git push → GitHub
```

- 工作区：在 VS Code 中编辑的文件，按 ⌘S 保存到本地磁盘。
- 暂存区：下一次提交的准备清单，也在本地，通常由 .git/index 保存。它记录上次 add 时选中的内容，不是另一个普通文件夹。
- 提交历史：commit 后形成可追踪的版本；当前分支向前移动。
- 远程仓库：push 后才接收到提交；本地 commit 不等于上传。

## 日常流程

在 MG2-RAG 项目根目录的终端中运行：

```bash
# 查看状态和未暂存的修改
git status
git diff

# 只选中这份笔记；其他笔记可替换文件路径
git add study-notes/01-git-vscode.md

# 检查暂存内容，再保存版本
git diff --cached
git commit -m "docs: update Git notes"

# 查看最近五次提交
git log --oneline -5
```

修改后要先按 ⌘S。add 后如果又修改文件，需要重新 add 才会将新内容加入本次提交。

## 分支与远程

```bash
git branch                 # 星号表示当前本地分支
git switch -c study/demo   # 创建并切换到新分支
git switch main            # 切换到已有主分支
git remote -v              # 查看远程地址，不会上传
```

分支名中的 / 是命名习惯，不代表文件夹。切换前检查并处理未提交修改。

本项目 upstream 指向作者仓库；个人 origin 配置及登录完成后，才使用：

```bash
git push -u origin study/day01-git
```

首次建立对应关系后，在这条分支通常可直接 git push。

## 常见状态

| 提示 | 意思 |
|---|---|
| Untracked files | 尚未纳入版本管理的新文件 |
| Changes not staged for commit | 修改尚未暂存 |
| Changes to be committed | 已暂存，等待提交 |
| working tree clean | 没有待提交改动，不代表已上传 |

撤回暂存选择，保留文件内容：

```bash
git restore --staged study-notes/01-git-vscode.md
```

实用习惯：提交前看差异；按文件 add；一次提交围绕一件事。忽略 .DS_Store，不提交密钥、.env、模型权重和大批数据。
