---
layout: archive
permalink: /readings/
title: readings
author_profile: false
sidebar:
  - image: "/images/about.jpg"
description: "Music makes i"
toc: true
og_image: "/images/about.jpg"
---

Music makes its best impression when it is shared by a friend over a story. Music is a medium for communicating and stimulating emotions, for

## Latest stories

<div class="grid__wrapper">
  {% assign collection = 'readings' %}
  {% assign posts = site[collection] | reverse %}
  {% for post in posts %}
    {% include archive-single.html type="grid" %}
  {% endfor %}
</div>
