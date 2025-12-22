{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}
   {% set ns = namespace(has_public_methods=false) %}
   {% for item in methods %}
       {% if not item.startswith('_') %}
           {% set ns.has_public_methods = true %}
       {% endif %}
   {% endfor %}
   {% if ns.has_public_methods %}
   .. rubric:: {{ _('Methods') }}

   .. autosummary::
   {% for item in methods %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block attributes %}
   {% set ns = namespace(has_public_attributes=false) %}
   {% for item in attributes %}
       {% if not item.startswith('_') %}
           {% set ns.has_public_attributes = true %}
       {% endif %}
   {% endfor %}
   {% if ns.has_public_attributes %}
   .. rubric:: {{ _('Attributes') }}

   .. autosummary::
   {% for item in attributes %}
      {%- if not item.startswith('_') %}
      ~{{ name }}.{{ item }}
      {%- endif -%}
   {%- endfor %}
   {% endif %}
   {% endblock %}
