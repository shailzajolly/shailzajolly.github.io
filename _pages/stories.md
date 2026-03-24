---
layout: page
permalink: /stories/
title: Stories
description: Being a silent observer, I notice the quiet details others walk past — this space is where those observations find their words.
nav: false
pagination:
  enabled: true
  collection: stories
  permalink: /page/:num/
  per_page: 5
  sort_field: date
  sort_reverse: true
  trail:
    before: 1
    after: 3
---

<ul class="post-list">
  {% if page.pagination.enabled %}
    {% assign postlist = paginator.posts %}
  {% else %}
    {% assign postlist = site.stories %}
  {% endif %}

{% for post in postlist %}
{% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
{% assign year = post.date | date: "%Y" %}

<li>
<h3>
<a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
</h3>
<p>{{ post.description }}</p>
<p class="post-meta">
{{ read_time }} min read &nbsp; &middot; &nbsp;
{{ post.date | date: '%B %d, %Y' }}
</p>
</li>
{% endfor %}

</ul>

{% if page.pagination.enabled %}
{% include pagination.liquid %}
{% endif %}
