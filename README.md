# RSS-GPT

[![](https://img.shields.io/github/last-commit/yinan-c/RSS-GPT/main?label=feeds%20refreshed)](https://yinan-c.github.io/RSS-GPT/)
[![](https://img.shields.io/github/license/yinan-c/RSS-GPT)](https://github.com/yinan-c/RSS-GPT/blob/master/LICENSE)

The workflow is still valid to use (using your own fork), but feeds update in this repo is suspended for now.
If you need a web GUI to manage feeds better, check out my latest project: [RSSBrew](https://github.com/yinan-c/RSSBrew), a self-hosted RSS-GPT alternative with more features and customizability, built with Django.

## What is this?

[Configuration Guide](https://yinan-c.github.io/rss-gpt-manual-en.html) | [中文简介](README-zh.md) | [中文教程](https://yinan-c.github.io/rss-gpt-manual-zh.html)

Using GitHub Actions to run a simple Python script repeatedly: Calling OpenAI API to generate summaries for RSS feeds, and push the generated feeds to GitHub Pages. Easy to configure, no server needed.

### Features

- Use ChatGPT to summarize RSS feeds, and attach summaries to the original articles, support custom summary length and target language.
- Aggregate multiple RSS feeds into one, remove duplicate articles, subscribe with a single address.
- Add filters to your own personalized RSS feeds.
- Host your own RSS feeds on GitHub repo and GitHub Pages.

![](https://i.imgur.com/7darABv.jpg)

## Quick configuration guide

- Fork this repo
- Add Repository Secrets
    - U_NAME: your GitHub username
    - U_EMAIL: your GitHub email
    - WORK_TOKEN: your GitHub personal access token with `repo` and `workflow` scope, get it from [GitHub settings](https://github.com/settings/tokens/new)
    - OPENAI_API_KEY(OPTIONAL, only needed when using AI summarization feature): Get it from [OpenAI website](https://platform.openai.com/account/api-keys)
- Enable GitHub Pages in repo settings, choose deploy from branch, and set the directory to `/docs`.
- Configure your RSS feeds in config.ini

You can check out [here](https://yinan-c.github.io/rss-gpt-manual-en.html) for a more detailed configuration guide.

## ChangeLog and updates

- As OpenAI released a new version of `openai` package on Nov 06, 2023.  [More powerful models are coming](https://openai.com/blog/new-models-and-developer-products-announced-at-devday), the way to call API also changed. As a result, the old script will no longer work with the latest version installed, and needs to be updated. Otherwise, you will have to set `openai==0.27.8` in `requirements.txt` to use the old version.
- Check out the [CHANGELOG.md](CHANGELOG.md).

### Contributions are welcome!

- Feel free to submit issues and pull requests.

## Support this project

- If you find it helpful, please star this repo. Please also consider buying me a coffee to help maintain this project and cover the expenses of OpenAI API while hosting the feeds. I appreciate your support.

<a href="https://www.buymeacoffee.com/yinan" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

## Example feeds being processed

These feeds on hosted in the [`docs/` subdirectory](https://github.com/yinan-c/RSS-GPT/tree/main/docs) in this repo as well as on my [GitHub Pages](https://yinan-c.github.io/RSS-GPT/). Feel free to subscribe in your favorite RSS reader.

I will consider hosting more feeds in the future. Email me or submit an issue if there are any questions using the script or any suggestions.

- https://www.alignmentforum.org/feed.xml?view=community-rss -> alignmentforum.xml
- https://www.lesswrong.com/feed.xml?view=frontpage-rss, https://www.lesswrong.com/feed.xml?view=frontpage-rss&karmaThreshold=30 -> lesswrong.xml
- https://kexue.fm/feed -> kexuefm.xml
- https://politepol.com/fd/uZeHlhOKLWF6.xml -> alignmentscience.xml
- https://politepol.com/fd/hvc24CgONayV.xml -> transformercircuits.xml
- https://politepol.com/fd/pix21PKyG4KK.xml -> grayswan.xml
- https://politepol.com/fd/JQseHIwNibhm.xml -> cais.xml
- https://politepol.com/fd/tEEQSjBKfPNk.xml, https://politepol.com/fd/HG65yVQWovKg.xml -> truthfulai.xml
- https://politepol.com/fd/6DNgCpV6Vl9l.xml -> transluce.xml
- https://politepol.com/fd/R5zvvYykxppD.xml -> aisi.xml
- https://politepol.com/fd/CugSKbvPdpjY.xml -> metr.xml
- https://politepol.com/fd/C3wOvys8aPKp.xml -> goodfire.xml
- https://papers.cool/arxiv/cs.AI/feed -> arxiv_csai.xml
- https://papers.cool/arxiv/cs.CV/feed -> arxiv_cscv.xml
- https://papers.cool/arxiv/cs.CL/feed -> arxiv_cscl.xml
- https://papers.cool/arxiv/cs.LG/feed -> arxiv_cslg.xml
- https://politepol.com/fd/e02qzlafmBbi.xml -> apollo.xml
