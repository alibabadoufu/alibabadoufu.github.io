---
layout: archive
permalink: /readings/
title: Paper Review
author_profile: true
description: "As a PhD candidate, I spend majority of my time on reading papers. This site will be used to document and summarize the papers that I have read."
toc: true
---

Albert Einstein — 'You do not really understand something unless you can explain it to your grandmother.'

## Latest stories

<div class="grid__wrapper">
  {% assign collection = 'readings' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
